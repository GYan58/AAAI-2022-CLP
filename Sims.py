from Settings import *
from Utils import *

class Client_Sim:
    def __init__(self,Loader,Model,Lr,wdecay,fixlr=False):
        self.TrainData = cp.deepcopy(Loader)
        self.Model = cp.deepcopy(Model)
        self.Wdecay = wdecay
        self.optimizer = torch.optim.SGD(self.Model.parameters(),lr=Lr,momentum=0.9,weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLR = fixlr
        self.DLen = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            self.DLen += len(inputs)

        self.train_loss = 0
        self.train_accu = 0
        self.gd_norm = 0
    
    def reload_data(self,loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def updateParas(self,Paras):
        self.Model.load_state_dict(Paras)
        
    def updateLR(self,lr):
        self.optimizer = torch.optim.SGD(self.Model.parameters(),lr=lr,momentum=0.9,weight_decay=self.Wdecay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        
    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR

    def selftrain(self,getnorm=False,btype=0):
        self.Model.train()
        train_loss = 0
        Accus = []
        grad_norm = []
        C = 0
        btype = int(btype)
        
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.Model(inputs)
            loss = self.loss_fn(outputs, targets)
            train_loss += loss
            loss.backward()
            self.optimizer.step()

            preds = outputs.argmax(dim=1)
            Ncorrect = torch.eq(preds,targets).sum().float().item() / len(inputs)
            Accus.append(Ncorrect)
            C += 1
            
            if getnorm == True:
                gnorm = 0
                for p in self.Model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    gnorm += param_norm.item() ** 2
                grad_norm.append(np.sqrt(gnorm))
                
            if btype > 0 and C >= btype:
                break
        
        self.train_loss = train_loss.item() / C
        self.train_accu = np.mean(Accus)
        if len(grad_norm) > 1:
            self.gd_norm = np.sum(grad_norm)
        
        if self.FixLR == False:
            self.scheduler.step()

        return self.train_loss, self.gd_norm

    def fim(self,loader=None,nout=10):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)

        self.Model.eval()
        Ts = []
        K = 5000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=100, shuffle=True)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=nout,
            variant="classif_logits",
            device="cuda"
        )
        
        Vec = PVector.from_model(self.Model)
        KL = F_Diag.vTMv(Vec).item()
        Tr = F_Diag.trace().item()

        return Tr,KL


#---------------------------------------------
class Server_Sim:
    def __init__(self,Loader,Model,Lr,wdecay=0,Fixlr=False):
        self.TrainData = cp.deepcopy(Loader)
        self.Model = cp.deepcopy(Model)
        self.optimizer = torch.optim.SGD(self.Model.parameters(),lr=Lr,momentum=0.9,weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loss = 0
        self.FixLr = Fixlr
        
    def reload_data(self,loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas
    
    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR

    def updateParas(self,Paras):
        self.Model.load_state_dict(Paras)

    def avgParas(self,Paras,Lens):
        Res = cp.deepcopy(Paras[0])
        Sum = np.sum(Lens)
        for ky in Res.keys():
            Mparas = 0
            for i in range(len(Paras)):
                Mparas += Paras[i][ky] * Lens[i] / Sum
            Res[ky] = Mparas

        return Res

    def aggParas(self,Paras,Lens):
        GParas = self.avgParas(Paras,Lens)
        self.updateParas(GParas)
        if self.FixLr == False:
            self.optimizer.step()
            self.scheduler.step()

    def evaluate(self,loader=None,max_samples=50000, verbose=True):
        self.Model.eval()

        loss, correct, samples, iters = 0, 0, 0, 0
        if loader == None:
            loader = self.TrainData
        C = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                
                _, preds = torch.max(y_.data, 1)
                
                loss += self.loss_fn(y_, y).item()
                correct += (preds == y).sum().item()
                samples += y_.shape[0]
                iters += 1

                if samples >= max_samples:
                    break

        return loss / iters, correct / samples
    
    def fim(self,loader=None,max_samples=50000,nout=10):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)

        self.Model.eval()
        Ts = []
        K = max_samples
        Trs = []
        KLs = []
        samples = 0
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) > K:
                    TLoader = torch.utils.data.DataLoader(dataset=Ts,batch_size=100,shuffle=True)
                    F_Diag = FIM(
                        model=self.Model,
                        loader=TLoader,
                        representation=PMatDiag,
                        n_output=nout,
                        variant="classif_logits",
                        device="cuda"
                    )
                    Tr = F_Diag.trace().item()
                    Trs.append(Tr)
                    Ts = []
                    
                    Vec = PVector.from_model(self.Model)
                    KL = F_Diag.vTMv(Vec).item()
                    KLs.append(KL)

                samples += len(x)
                if samples >= max_samples:
                    break
        
        Tr = np.mean(Trs)
        KL = np.mean(KLs)
        return Tr,KL
