import argparse
import signal
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch, torchvision
from datasets import CarOrientationDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))





# Define the model
class Model(torch.nn.Module):
    def __init__(self, N):
        super(Model, self).__init__()
        self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        base_model_output_size = list(self.base_model.parameters())[-1].size(0)
        self.preds = torch.nn.Linear(base_model_output_size, N)


    def forward(self, images):
        image_features = self.base_model(images)
        preds = self.preds(torch.squeeze(image_features))
        return torch.squeeze(preds)





class OrientationLoss(torch.nn.Module):
    '''
    Loss function combining categorical cross entropy + L1 (to penalize 180 deg mistakes)

    Parameters
    ----------
    N : Number of classes
    lambda : the weight to put on the regression loss

    Forward-pass Parameters
    -----------------------
    x : The outputs of the model - an N-way classification
                                   0 = [0°, N/360°), ..., N = [(N-1)/360°, 0°)
    y : The actual orientation (0 to N)

    Returns
    -------
    Loss : The sum of the losses
    '''
    def __init__(self, N, _lambda):
        super(OrientationLoss, self).__init__()
        self.class_loss = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.num_classes = N
        self._lambda = _lambda

    def forward(self, x, y):
        diff = torch.abs(torch.argmax(x,1) - y)
        l1_loss = torch.min(diff, (diff - self.num_classes)*-1).type(torch.cuda.FloatTensor) / (self.num_classes/2)
        class_loss = self.class_loss(x, y)
        return torch.mean(l1_loss) * self._lambda + class_loss





def train_model(model, dataloaders, loss_func, optimizer, scheduler, experiment, num_epochs=100):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ["train", "val", "test"]:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            # Create dataset iterator
            stat_dict = {"Epoch": epoch}
            running_losses = []
            results = {
                'image': [],
                'orientation': [],
                'predicted_orientation': [],
                'source': [],
            }
            iterator = tqdm(dataloaders[phase], postfix=stat_dict, ncols=115, desc=phase)

            for data in iterator:
                images = data['image'].to(DEVICE)
                orientations = data['orientation'].to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, orientations)

                    # Backward pass + Optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Compute & Display statistics
                    running_losses.append(loss.item())
                    results['image'].extend(data['image_file'])
                    results['orientation'].extend(data['orientation'].numpy())
                    results['predicted_orientation'].extend(preds.cpu().numpy())
                    results['source'].extend(data['source'])
                    stat_dict['Loss'] = "{:.5f}".format(np.mean(running_losses))
                    stat_dict['Acc'] = "{:.5f}".format(100*np.mean(np.equal(results['orientation'], results['predicted_orientation'])))
                    iterator.set_postfix(stat_dict)

            # Save trained weights
            if phase == "train":
                torch.save(model.state_dict(), "models/epoch_{:02d}_experiment_{:02d}.pyt".format(epoch,experiment))

            # Save outputs
            if phase in ["val", "test"]:
                pd.DataFrame(results).to_csv("outputs/{}_epoch_{:02d}_experiment_{:02d}.csv".format(phase,epoch,experiment), index=False)




def test_model(model, test_dataloader):
    model.load_state_dict(torch.load("/home/mike/Projects/DIVA/orientation_estimation/models/epoch_07.pyt"))
    model.eval()

    # Create dataset iterator
    stat_dict = {}
    results = {
        'image': [],
        'predicted_orientation': [],
        'source': [],
    }
    iterator = tqdm(test_dataloader, postfix=stat_dict, ncols=115, desc="Testing")

    for data in iterator:
        images = data['image'].to(DEVICE)

        with torch.set_grad_enabled(False):
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Compute & Display statistics
            results['image'].extend(data['image_file'])
            results['predicted_orientation'].extend(preds.cpu().numpy())
            results['source'].extend(data['source'])
            iterator.set_postfix(stat_dict)

    pd.DataFrame(results).to_csv("outputs/saved_model_outputs.csv", index=False)





def main(N, _lambda, batch_size, workers, experiment, data):
    # Create model
    model = Model(N).to(DEVICE)

    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            CarOrientationDataset("train", data), batch_size=batch_size, shuffle=True, num_workers=workers,
        ),
        'val': torch.utils.data.DataLoader(
            CarOrientationDataset("val"), batch_size=batch_size, shuffle=False, num_workers=workers,
        ),
        'test': torch.utils.data.DataLoader(
            CarOrientationDataset("test"), batch_size=batch_size, shuffle=False, num_workers=workers,
        ),
        # 'diva': torch.utils.data.DataLoader(
        #     CarOrientationDataset("diva") batch_size=batch_size, shuffle=False, num_workers=workers,
        # ),
    }

    # Create loss_func
    # loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)
    loss_func = OrientationLoss(N, _lambda).to(DEVICE)

    # Create optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model, dataloaders, loss_func, optimizer, scheduler, experiment)
    # test_model(model, dataloaders["test"])





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a network to predict car orientation")
    parser.add_argument(
        "--num-bins", '-n', dest="bins",
        default=36,
        type=int,
        help="Number of classification bins",
    )
    parser.add_argument(
        "--batch-size", '-b', dest="batch_size",
        default=256,
        type=int,
        help="Batch size of images",
    )
    parser.add_argument(
        "--lambda", '-l', dest="_lambda",
        default=1.0,
        type=float,
        help="Regularization weight on l1 loss",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        dest="workers",
        help="Number of worker processes",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        help="Experiment number",
    )
    parser.add_argument(
        "--data",
        default="all_data",
        help="How to filter the training data",
    )
    args = parser.parse_args()
    main(args.bins, args._lambda, args.batch_size, args.workers, args.experiment, args.data)
