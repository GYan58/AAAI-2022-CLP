from Settings import *
from Models import CNN as C
from Models import ResNet as R
from Models import VGG as V


def load_Model(Type, Name):
    Model = None
    if Type == "cnn":
        if Name == "mnist":
            Model = C.cnn_mnist()
    
        if Name == "cifar10":
            Model = C.cnn_cifar10()
        
        if Name == "cifar100":
            Model = C.cnn_cifar100()

    if Type == "vgg":
        if Name == "mnist":
        Model = V.vgg_mnist()
    
        if Name == "cifar10":
            Model = V.vgg_cifar10()
        
        if Name == "cifar100":
            Model = V.vgg_cifar100()

    if Type == "resnet":
        if Name == "mnist":
            Model = R.resnet_mnist()
    
        if Name == "cifar10":
            Model = R.resnet_cifar10()
            
        if Name == "cifar100":
            Model = R.resnet_cifar100()


def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data",train=True,download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data",train=False,download=True)
    TrainX, TrainY = data_train.data.transpose((0,3,1,2)),np.array(data_train.targets)
    TestX, TestY= data_test.data.transpose((0,3,1,2)),np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY
    
def get_cifar100():
    data_train = torchvision.datasets.CIFAR100(root="./data",train=True,download=True)
    data_test = torchvision.datasets.CIFAR100(root="./data",train=False,download=True)
    TrainX, TrainY = data_train.data.transpose((0,3,1,2)),np.array(data_train.targets)
    TestX, TestY= data_test.data.transpose((0,3,1,2)),np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_mnist():
    data_train = torchvision.datasets.MNIST(root="./data",train=True,download=True)
    data_test = torchvision.datasets.MNIST(root="./data",train=False,download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.targets)
    TestX, TestY= data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

# add Blur
class Addblur(object):

    def __init__(self, blur="Gaussian"):
        self.blur= blur

    def __call__(self, img):
        if self.blur == "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        if self.blur == "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        if self.blur == "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img

# add noise
class AddNoise(object):
    def __init__(self, noise="Gaussian"):
        self.noise = noise
        # pepper
        self.density = 0.8
        # Gaussian
        self.mean = 0.0
        self.variance = 10.0
        self.amplitude = 10.0

    def __call__(self, img):

        img = np.array(img) #
        h, w, c = img.shape

        if self.noise == "pepper":
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      
            mask = np.repeat(mask, c, axis=2)                                               
            img[mask == 2] = 0                                                              
            img[mask == 1] = 255                                                            

        if self.noise == "Gaussian":
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255

        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img


def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=False, verbose=False, balancedness=0.0):
    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    if balancedness >= 1.0:
        data_per_client = [n_data // n_clients] * n_clients
        data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    else:
        fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1 / n_clients + (1 - 0.1) * fracs
        fracs = []
        Sum = n_clients * (n_clients + 1) / 2
        SProb = 0
        for i in range(n_clients-1):
            prob = int((i+1) / Sum * 100) / 100
            SProb += prob
            fracs.append(prob)
        
        Left = 1 - SProb
        fracs.append(Left)
        bfrac = 0.5 / n_clients
        for i in range(len(fracs)):
            fracs[i] = fracs[i] / 2.0 + bfrac
        print(fracs,np.sum(fracs))
        
        data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]

        data_per_client = data_per_client[::-1]

        data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    CCount = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = CCount % n_labels
        CCount += 1
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
            
            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels
        
        Ls = labels[client_idcs]
        Ds = data[client_idcs]
        Xs = []
        Ys = []
        Datas = {}
        for k in range(len(Ls)):
            L = Ls[k]
            D = Ds[k]
            if L not in Datas.keys():
                Datas[L] = [D]
            else:
                Datas[L].append(D)
        
        Kys = list(Datas.keys())
        Kl = len(Kys)
        CT = 0
        k = 0
        while CT < len(Ls):
            Id = Kys[k%Kl]
            k += 1
            if len(Datas[Id]) > 0:
                Xs.append(Datas[Id][0])
                Ys.append(Id)
                Datas[Id] = Datas[Id][1:]
                CT += 1
        
        clients_split += [(np.array(Xs), np.array(Ys))]
        del Xs, Ys, Kys
        gc.collect()
        

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    print(len(clients_split[0][0]))

    return clients_split


def get_train_data_transforms(name,aug=False,blur=False,noise=False,normal=False):
    Ts = [transforms.ToPILImage()]
    if name == "mnist":
        Ts.append(transforms.Resize((32, 32)))

    if aug == True and name == "cifar10":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())

    if blur == True:
        Ts.append(Addblur())

    if noise == True:
        Ts.append(AddNoise())

    Ts.append(transforms.ToTensor())
    
    if normal == True:
       print("*Train Normalization!")
       if name == "mnist":
           Ts.append(transforms.Normalize((0.06078,),(0.1957,)))
       if name == "cifar10":
           Ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
       if name == "cifar100":
           Ts.append(transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761)))

    return transforms.Compose(Ts)

def get_test_data_transforms(name,normal=False):
    transforms_eval_F = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
    }
    
    transforms_eval_T = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,),(0.1957,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761))
        ]),
    }
    
    if normal == False:
        return transforms_eval_F[name]
    else:
        print("*Test Normalization!")
        return transforms_eval_T[name]


class CustomImageDataset(Dataset):
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]


def get_loaders(Name, n_clients=10, classes_per_client=10, aug=False, part=False, noise=False, blur=False,normal=False,prate=1.0,balance=True,dshuffle=True,batchsize=128):
    TrainX, TrainY, TestX, TestY = [],[],[],[]
    if Name == "mnist":
        TrainX, TrainY, TestX, TestY = get_mnist()
    if Name == "cifar10":
        TrainX, TrainY, TestX, TestY = get_cifar10()
    if Name == "cifar100":
        TrainX, TrainY, TestX, TestY = get_cifar100()

    transforms_train = get_train_data_transforms(Name,aug,blur,noise,normal)
    transforms_eval = get_test_data_transforms(Name,normal)

    inb = 0.0
    if balance == True:
        inb = 1.0

    splits = split_image_data(TrainX, TrainY, n_clients, classes_per_client,True,True,inb)

    client_loaders = []
    if part == False:
        for x, y in splits:
            SumL += len(x)
            client_loaders.append(torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),batch_size=batchsize, shuffle=dshuffle))
    else:
        for x, y in splits:
            L = int(len(x) * prate)
            SumL += len(x[:L])
            client_loaders.append(torch.utils.data.DataLoader(CustomImageDataset(x[:L], y[:L], transforms_train),batch_size=batchsize, shuffle=dshuffle))

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(TrainX, TrainY, transforms_eval), batch_size=128,shuffle=True)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(TestX, TestY, transforms_eval), batch_size=128,shuffle=True)
    stats = {"split": [x.shape[0] for x, y in splits]}

    return client_loaders, train_loader, test_loader, stats
