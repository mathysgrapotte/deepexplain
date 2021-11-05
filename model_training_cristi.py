import torch
import torch.nn as nn

n_epochs = 2 # Dataset is quite large, 2 epochs are enough for this simple net
learning_rate = 0.01
momentum = 0.5
log_interval = 100

loss_func = nn.CrossEntropyLoss()

def train(model, train_loader, test_loader):
    test(model, test_loader)
    for epoch in range(1, n_epochs + 1):
      train_step(model, train_loader, epoch)
      test(model, test_loader)

def train_step(model, train_loader, epoch):
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum)
  model.train()
  for batch_idx, (data, label) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      
def test(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, label in test_loader:
      output = model(data)
      test_loss += loss_func(output, label).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(label.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))