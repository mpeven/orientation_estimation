import signal
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch, torchvision
from datasets import CarOrientationDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))



def train_model(model, dataloaders, loss_func, optimizer, scheduler, num_epochs=100):
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
                torch.save(model.state_dict(), "models/epoch_{:02d}.pyt".format(epoch))

            # Save outputs
            if phase in ["val", "test"]:
                pd.DataFrame(results).to_csv("outputs/{}_epoch_{:02d}.csv".format(phase,epoch), index=False)




class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        base_model_output_size = list(self.base_model.parameters())[-1].size(0)
        self.preds = torch.nn.Linear(base_model_output_size, 36)


    def forward(self, images):
        image_features = self.base_model(images)
        preds = self.preds(torch.squeeze(image_features))
        return torch.squeeze(preds)



def main():
    # Create model
    model = Model().to(DEVICE)

    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(CarOrientationDataset("train"), batch_size=64, shuffle=True, num_workers=8),
        'val':   torch.utils.data.DataLoader(CarOrientationDataset("val"), batch_size=64, shuffle=False, num_workers=8),
        'test':  torch.utils.data.DataLoader(CarOrientationDataset("test"), batch_size=64, shuffle=False, num_workers=8)
    }

    # Create loss_func
    loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

    # Create optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model, dataloaders, loss_func, optimizer, scheduler)




if __name__ == '__main__':
    main()
