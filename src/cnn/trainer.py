import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm, tqdm_notebook
from torch.optim import lr_scheduler

DEVICE = torch.device("cuda")

def fit_epoch(model, train_loader, criterion, optimazer, scheduler):
  """ Обучение модели  c настройкой весов"""
  
  scheduler.step()
  running_loss = 0.0
  running_corrects = 0
  processed_data = 0

  # Итерация по бачам
  for inputs, labels in train_loader:
    # Обертка переменных для забуска на видео карте
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    # Инициализируем градиенты параметров
    optimazer.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # backward pass и оптимизация
    loss.backward()
    optimazer.step()
    preds = torch.argmax(outputs, 1)

    # Подсчет статистики 
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    processed_data += inputs.size(0)

  train_loss = running_loss / processed_data
  train_acc = running_corrects.cpu().numpy() / processed_data

  return train_loss, train_acc

def eval_epoch(model, val_loader, criterion):
  """ Этап валидации для промежуточной оценки качества обучения модели"""

  model.eval()
  running_loss = 0.0
  running_corrects = 0
  processed_size = 0

  # Итерация по бачам
  for inputs, labels in val_loader:
      # Обертка переменных для забуска на видео карте
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)

      # Проходимся по всей моедли, но уже не корректируем веса
      with torch.set_grad_enabled(False):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, 1)

      # Подсчет статистики 
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)
      processed_size += inputs.size(0)

  val_loss = running_loss / processed_size
  val_acc = running_corrects.double() / processed_size

  return val_loss, val_acc

def train(train_dataset, val_dataset, model, epochs, batch_size):
  """ Обучение модели на батчах с определенным количеством эпох """

  # Разделение множества данных на батчи
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  best_model_wts = model.state_dict()
  best_acc = 0.0

  history = []
  log_template =  "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
  val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

  with tqdm(desc="epoch", total=epochs) as pbar_outer:
    # Оптимизационный алгоритм (подсчет и изменения градиента)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(params=[
            {"params": model.fc.parameters()},
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.layer3.parameters(), "lr": 1e-4},
            {"params": model.layer2.parameters(), "lr": 1e-4},
            {"params": model.layer1.parameters(), "lr": 1e-4},
        ],
        lr=1e-3
    )

    # Метод вычисления точности результата (доствоверности), оптимизируемая функция
    criterion = nn.CrossEntropyLoss()

    # Уменьшение шага градиента в 10 раз
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    # Итерируемся по эпохам 
    for epoch in range(epochs):
      # Обучение
#       if epoch == (epochs // 2):
#             model.unfreeze()
        
      train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, \
                                        exp_lr_scheduler)

      # Проверка
      val_loss, val_acc = eval_epoch(model, val_loader, criterion)
      
      # Если достиглось лучшее качество, то запомним веса модели
      if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = model.state_dict()

      history.append((train_loss, train_acc, val_loss, val_acc))
      pbar_outer.update(1)
      tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss, \
                                     v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
  print(f'Best val accurancy: {best_acc}')
  model.load_state_dict(best_model_wts)

  return model, history

def predict(model, test_loader):
  """ Тестирование и измерение точности модели """

  with torch.no_grad():
    logits = []

    for inputs in test_loader:
      inputs = inputs.to(DEVICE)
      model.eval()
      outputs = model(inputs).cpu()
      logits.append(outputs)

  probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
  return probs

# def imshow(inp, title=None, plt_ax=plt, default=False):
#     """ Imshow для тензоров """
  
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt_ax.imshow(inp)

#     if title is not None:
#         plt.title(title)
#     plt_ax.grid(False)

# def show_imgs(val_dataset, train_dataset):
#     random_characters = int(np.random.uniform(0, 1406))
#     im_train, label = val_dataset[random_characters]
#     fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(10, 10), \
#                             sharey=True, sharex=True)
#     for fig_x in ax.flatten():
#         random_characters = int(np.random.uniform(0,1406))
#         im_train, label = train_dataset[random_characters]
#         img_label = " ".join(map(lambda x: x.capitalize(),\
#                     train_dataset.label_encoder.inverse_transform([label])[0].split('_')))
#         imshow(im_train.data.cpu(), \
#             title=img_label,plt_ax=fig_x)