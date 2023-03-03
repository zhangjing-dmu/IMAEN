import argparse
import torch.nn as nn
# from models.gcn import GCNNet
# from models.GranphConv import GraphConvNet
from models.lagcn_mcnn import GCNNet
# from pretrained_model.gcn import GCNNet
from utils import *
from cvae_models import VAE
from plot import pred_scatter_plot

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, vae):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        # X_list = get_augmented_features(args.concat)
        optimizer.zero_grad()
        output = model(data, vae)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        # ind_sorted = np.argsort(loss.cpu().data).cuda()  # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
        # loss_sorted = loss[ind_sorted]
        #
        # remember_rate = 1 - 0.3
        # num_remember = int(remember_rate * len(loss_sorted))
        #
        # ind_update = ind_sorted[:num_remember]
        # loss = loss_fn1(output[ind_update], data.y[ind_update].view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader, vae):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data, vae)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

# loss_fn = nn.MSELoss(reduce=False)  # reduce = False，返回向量形式的 loss　
# loss_fn1 = nn.MSELoss()
loss_fn = nn.MSELoss()
LOG_INTERVAL = 20

def main(args):
  dataset = args.dataset
  modeling = [GCNNet]
  # modeling = [GraphConvNet]
  model_st = modeling[0].__name__

  cuda_name = "cuda:0"
  print('cuda_name:', cuda_name)

  TRAIN_BATCH_SIZE = args.batch_size
  TEST_BATCH_SIZE = args.batch_size
  LR = args.lr
  
  NUM_EPOCHS = args.epoch

  print('Learning rate: ', LR)
  print('Epochs: ', NUM_EPOCHS)

  # Main program: iterate over different datasets
  print('\nrunning on ', model_st + '_' + dataset )
  processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
  processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
  if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
     print('please run create_data.py to prepare data in pytorch format!')
  else:
    train_data = TestbedDataset(root='data', dataset=dataset+'_train')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test')
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False,drop_last=True)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    # model = modeling[0](k1=1,k2=2,k3=3,embed_dim=128, num_layer=1, device=device).to(device)
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes).to(device)
    model = modeling[0](k1=1,k2=2,k3=3,embed_dim=128, num_layer=1, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    #model_file_name = 'model' + model_st + '_' + dataset +  '.model'
    result_file_name = 'result' + model_st + '_' + dataset +  '.csv'

    for epoch in range(NUM_EPOCHS):
      train(model, device, train_loader, optimizer, epoch+1, vae)
      G,P = predicting(model, device, test_loader, vae)
      ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G.reshape(G.shape[0],-1),P.reshape(P.shape[0],-1))]
      if ret[1]<best_mse:
        if args.save_file:
          model_file_name = args.save_file + '.model'
          torch.save(model.state_dict(), model_file_name)
        
        
        with open(result_file_name,'w') as f:
          f.write('rmse,mse,pearson,spearman,ci,rm2\n')
          f.write(','.join(map(str, ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_ci = ret[-2]
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
      else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
      # if epoch == 999:
      #   scatter = pred_scatter_plot(G, P, 'Davis Dataset: Predictions vs True Values', 'True Values',
      #                               'Predictions',
      #                               True, 'cc')
      #   print(scatter)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run DeepGLSTM")

  parser.add_argument("--dataset", type=str, default='Metz', help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")

  parser.add_argument("--epoch",
                      type=int,
                      default=1000,
                      help="Number of training epochs. Default is 1000."
                      ) 
  
  parser.add_argument("--lr",
                      type=float,
                      default=0.0005,
                      help="learning rate",
                      )
  
  parser.add_argument("--batch_size", type=int,
                      default=256,
                      help="Number of drug-tareget per batch. Default is 128 for davis.")   # batch 128 for Davis
  
  parser.add_argument("--save_file", type=str,
                      default='./pretrained_model/Metz.model',
                      help="Where to save the trained model. For example davis.model")
  parser.add_argument("--concat", type=int, default=1)
  parser.add_argument("--encoder_layer_sizes", type=list, default=[128, 256])
  # parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 128])
  parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 78])
  parser.add_argument("--latent_size", type=int, default=10)
  args = parser.parse_args()
  print(args)
  main(args)