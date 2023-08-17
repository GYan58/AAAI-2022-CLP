from Settings import *
from Utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class FL_Proc:
    def __init__(self, configs, model):
        self.Name = configs["name"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"]
        self.PClients = configs["pclients"]
        self.PClass = configs["pclass"]
        self.BRounds = configs["bround"]
        self.Aug = configs["aug"]
        self.MaxIter = configs["iters"]
        self.LogStep = configs["logstep"]
        self.DataType = configs["dtype"]
        self.LR = configs["learning_rate"]
        self.Normal = configs["normal"]
        self.IDType = configs["itype"]
        self.PRate = configs["prate"]
        self.FixLR = configs["fixlr"]
        self.WDecay = configs["wdecay"]
        self.Balance = configs["balance"]
        self.CheckMetric = configs["check_metric"]
        self.DShuffle = configs["data_shuffle"]
        self.BatchSize = configs["batch_size"]
        self.GlobalLR = configs["global_lr"]
        self.ClientBSize = configs["nbatch"]
        self.IDRange = configs["id_range"]
        
        # name variables to store corrsponding dataset
        self.BModel = load_Model(self.Name, self.ModelName)
        self.CLoaders = None
        self.TrainLoader = None
        self.TestLoader = None

        self.PLoaders = None
        self.PTrainLoader = None
        self.PTestLoader = None
        
        self.ALoaders = None
        self.ATrainLoader = None
        self.ATestLoader = None

        self.NLoaders = None
        self.NTrainLoader = None
        self.NTestLoader = None

        self.BLoaders = None
        self.BTrainLoader = None
        self.BTestLoader = None

    def get_ori_datas(self):
        self.CLoaders, self.TrainLoader, self.TestLoader, Stat = get_loaders(self.Name, self.NClients,self.PClass,self.Aug,False,False,False,self.Normal,1.0,self.Balance,self.DShuffle,self.BatchSize)
        
    def get_aug_datas(self):
        self.ALoaders, self.ATrainLoader, self.ATestLoader, Stat = get_loaders(self.Name, self.NClients,self.PClass,True,False,False,False,self.Normal,1.0,self.Balance,self.DShuffle,self.BatchSize)

    def get_part_datas(self):
        print("PRate:",self.PRate)
        self.PLoaders, self.PTrainLoader, self.PTestLoader, Stat = get_loaders(self.Name, self.NClients,self.PClass,self.Aug,True,False,False,self.Normal,self.PRate,self.Balance,self.DShuffle,self.BatchSize)

    def get_noise_datas(self):
        self.NLoaders, self.NTrainLoader, self.NTestLoader, Stat = get_loaders(self.Name, self.NClients,self.PClass,self.Aug,False,True,False,self.Normal,1.0,self.Balance,self.DShuffle,self.BatchSize)

    def get_blur_datas(self):
        self.BLoaders, self.BTrainLoader, self.BTestLoader, Stat = get_loaders(self.Name, self.NClients,self.PClass,self.Aug,False,False,True,self.Normal,1.0,self.Balance,self.DShuffle,self.BatchSize)

    def main(self):
        ## number of outputs
        self.NOut = 10
        DTPEs = self.DataType.split("_")
        DFirst = DTPEs[0]
        DSecond = DTPEs[1]
        
        ITPEs = self.IDType.split("_")
        IFirst = ITPEs[0]
        ISecond = ITPEs[1]
        
        BSs = self.ClientBSize.split("_")
        BFirst = BSs[0]
        BSecond = BSs[1]
        
        RTPEs = self.IDRange.split("_")
        RFirst = int(float(RTPEs[0]) * self.NClients)
        RSecond = int(float(RTPEs[1]) * self.NClients)
        
        SelType = IFirst
        BType = BFirst
        
        if "ori" in DTPEs:
            print("Get Origin Data...")
            self.get_ori_datas()
        if "part" in DTPEs:
            print("Get Part Data...")
            self.get_part_datas()
        if "noise" in DTPEs:
            print("Get Noise Data...")
            self.get_noise_datas()
        if "blur" in DTPEs:
            print("Get Blur Data...")
            self.get_blur_datas()
        if "aug" in DTPEs:
            print("Get Aug Data...")
            self.get_aug_datas()
            
        Clients = {}
        IDs = []
        for c in range(self.NClients):
            IDs.append(c)

        Server = None
        if DFirst == "ori":
            Server = Server_Sim(self.TrainLoader,self.BModel,self.LR,self.WDecay,self.FixLR)
            print("***Init Origin Data***")
            for c in range(self.NClients):
                Clients[c] = Client_Sim(self.CLoaders[c],self.BModel, self.LR, self.WDecay, self.FixLR)
                
        if DFirst == "part":
            Server = Server_Sim(self.PTrainLoader,self.BModel,self.LR,self.WDecay,self.FixLR)
            print("***Init Part Data***")
            for c in range(self.NClients):
                Clients[c] = Client_Sim(self.PLoaders[c], self.BModel, self.LR, self.WDecay, self.FixLR)
                
        if DFirst == "noise":
            Server = Server_Sim(self.NTrainLoader,self.BModel,self.LR,self.WDecay,self.FixLR)
            print("***Init Noise Data***")
            for c in range(self.NClients):
                Clients[c] = Client_Sim(self.NLoaders[c], self.BModel, self.LR, self.WDecay, self.FixLR)
                
        if DFirst == "blur":
            Server = Server_Sim(self.BTrainLoader,self.BModel,self.LR,self.WDecay,self.FixLR)
            print("***Init Blur Data***")
            for c in range(self.NClients):
                Clients[c] = Client_Sim(self.BLoaders[c], self.BModel, self.LR, self.WDecay, self.FixLR)
                
        if DFirst == "aug":
            Server = Server_Sim(self.ATrainLoader,self.BModel,self.LR,self.WDecay,self.FixLR)
            print("***Init Aug Data***")
            for c in range(self.NClients):
                Clients[c] = Client_Sim(self.ALoaders[c], self.BModel, self.LR, self.WDecay, self.FixLR)
           
        MaxIters = self.MaxIter
        C = 0
        
        FIDs = []
        F1 = []
        F2 = []
        for i in range(int(self.NClients/2)):
            F1.append(i)
            F2.append(self.NClients - 1 -i)
        
        for i in range(len(F1)):
            FIDs.append(F1[i])
            FIDs.append(F2[i])
            
        print("FIDs:",FIDs,len(IDs),len(FIDs))
        if len(FIDs) < len(IDs):
            FIDs = IDs
            
        DLs = {}
        for c in IDs:
            DLs[c] = Clients[c].DLen
        
        Recover = False
        CommSize = 0
        CommTime = 0
        RIDs = FIDs[:RFirst]
        LastIDs = RIDs[:10]
        
        for i in range(MaxIters):
            C += 1

            if i % self.LogStep == 0:
                # get the logged results
                teloss, teaccu = Server.evaluate(self.TestLoader,max_samples=10000)
                print("*test loss and accuracy:", teloss, teaccu)
                
                SLs = []
                GNs = []
                for ky in LastIDs:
                    traccu = Clients[ky].train_accu
                    trloss = Clients[ky].train_loss
                    trlr = Clients[ky].getLR()
                    gdnorm = Clients[ky].gd_norm
                    SLs.append(trloss)
                    GNs.append(gdnorm)
                
                if self.CheckMetric == True:    
                    if self.BRounds >= 0:
                        GParas = Server.getParas()
                        for ky in LastIDs:
                            Clients[ky].updateParas(GParas)
                            gtrace,kl = Clients[ky].fim(nout=self.NOut)

            CollectParas = []
            DLens = []

            if i >= self.BRounds and Recover == False:
                Recover = True
                SelType = ISecond
                BType = BSecond
                RIDs = FIDs[:RSecond]
                
                if DSecond == "ori":
                    print("*Recover Origin Data...")
                    for c in range(self.NClients):
                        Clients[c].reload_data(self.CLoaders[c])

                if DSecond == "part":
                    print("*Recover Partial Data...")
                    for c in range(self.NClients):
                        Clients[c].reload_data(self.PLoaders[c])

                if DSecond == "noise":
                    print("*Recover Noise Data...")
                    for c in range(self.NClients):
                        Clients[c].reload_data(self.NLoaders[c])
                
                if DSecond == "blur":
                    print("*Recover Blur Data...")
                    for c in range(self.NClients):
                        Clients[c].reload_data(self.BLoaders[c])
                 
                if DSecond == "aug":
                    print("*Recover Aug Data...")
                    for c in range(self.NClients):
                        Clients[c].reload_data(self.ALoaders[c])

            GIDs = None
            if SelType == "random":
                GIDs = rd.sample(RIDs, self.PClients)
            if SelType == "former":
                GIDs = RIDs[:self.PClients]
            if SelType == "latter":
                GIDs = RIDs[-self.PClients:]
            if SelType == "all":
                GIDs = RIDs

            LastIDs = GIDs
            for c in GIDs:
                GParas = Server.getParas()
                Clients[c].updateParas(GParas)
                if self.GlobalLR == True:
                    GLr = Server.getLR()
                    Clients[c].updateLR(GLr)
                
                if (i + 1) % self.LogStep == 0:
                    Clients[c].selftrain(getnorm=True,btype=BType)
                else:
                    Clients[c].selftrain(btype=BType)
                
                CollectParas.append(Clients[c].getParas())
                DLens.append(Clients[c].DLen)
            
            if len(CollectParas) > 0:
                Server.aggParas(CollectParas,DLens)
            else:
                print("Error...")
                
                
               
if __name__ == '__main__':
    Configs = {}
    Configs['name'] = "cifar10"
    Configs["mname"] = "vgg"
    Configs['nclients'] = 64
    Configs['pclients'] = 8
    Configs['pclass'] = 5
    Configs['logstep'] = 1
    Configs["dtype"] = "ori"
    Configs["res_data"] = True
    Configs["dtype"] = "part_ori" # data types
    Configs["itype"] = "random_random" # the method of selecting clients
    Configs["nbatch"] = "0_0" # the number of used batches at each training, if 0, use all batches
    Configs["id_range"] = "1.0_1.0"
    Configs["check_metric"] = False
    Configs["data_shuffle"] = True
    Configs["aug"] = True
    Configs["normal"] = True
    Configs["fixlr"] = False
    Configs["balance"] = True
    Configs["global_lr"] = False 
    Configs["prate"] = 0.1
    Configs["bround"] = 0
    Configs["learning_rate"] = 0.01
    Configs["wdecay"] = 1e-5
    Configs["batch_size"] = 16
    Configs["iters"] = 200
                                    
    FLSim = FL_Proc(Configs,Model)
    FLSim.main()

