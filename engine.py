# information back propagation
import torch
from algo import LinearEquationSolver
# from algo_decimal import LinearEquationSolver
import numpy as np
import torch.nn as nn

class Backprop:
    def __init__(self, low_scope=-1, up_scope=1, low_bound=None, high_bound=None, epsilon=0, repeat_times=20000):
        self.up_scope = up_scope
        self.low_scope = low_scope

        self.low = low_bound
        self.high = high_bound
        self.epsilon = epsilon
        if self.high is not None and self.low is not None:
            assert self.high - self.epsilon > self.low + self.epsilon

        self.repeat_times = repeat_times

    def _pick_one(self, special_solution, null_solution):
        v = np.random.uniform(self.low_scope, self.up_scope, (1, len(null_solution)))
        return v.dot(null_solution).flatten() + special_solution
    
    def _check_whether_in_range(self, v):
        if self.low is None and self.high is None:
            return True
        if self.low is None:
            j = v <= self.high - self.epsilon
            if j.all():
                return True
            return False
        if self.high is None:
            j = v >= self.low + self.epsilon
            if j.all():
                return True
            return False
        j = np.logical_and(v >= self.low + self.epsilon, v <= self.high - self.epsilon)
        if v.all():
            return True
        return False

    def random_pick_solutions(self, sols, N):
        #
        special_solution = sols["special_solution"]
        null_solution = sols["null_solution"]
        # condition
        if special_solution is None:
            raise RuntimeError("Passed value has no solution")
        if null_solution == []:
            return [special_solution]
        # stack
        null_solution = np.stack(null_solution)
        # generate N random solution within the required range
        N_samples = []
        for _ in range(N):
            s = self._pick_one(special_solution, null_solution)
            if not self._check_whether_in_range(s):
                count = 0
                for _ in range(self.repeat_times):
                    s = self._pick_one(special_solution, null_solution)  
                    if self._check_whether_in_range(s):
                        N_samples.append(s)
                        break
                    count += 1
                if count == self.repeat_times:
                    print(special_solution)
                    print(s)
                    raise RuntimeError("Max repeat times exceeded. Condider changing scope")
            else:
                N_samples.append(s)
        return N_samples
    
class Mutiple_Back:
    def __init__(self, back_list):
        self.back_list = back_list

        self.all_values = []   # for test purpose

    def backprop(self, value):
        value_list = [value]
        for id, layer in enumerate(self.back_list):
            print(f"Backpropagating layer {len(self.back_list)-id}")
            new_value_list = []
            for value in value_list:
                back_value = layer.backprop(value)
                new_value_list += back_value
            value_list = new_value_list

            self.all_values.append(value_list)    # for test purpose

        return value_list

class Back_Softmax(Backprop):
    def __init__(self, N,  epsilon_ln=0.0001):
        super().__init__(low_bound=0)

        self.N = N

        self.epsilon_ln = epsilon_ln   # could adjust to a smaller value

    def backprop(self, value):
        #
        dim = value.shape[-1]
        mat = []
        for ind, v in enumerate(value):
            t = v * torch.ones((dim,))
            t[ind] = t[ind] - 1
            mat.append(t)
        mat = torch.stack(mat).numpy()
        b = np.zeros((dim, 1))
        sols = LinearEquationSolver(mat, b).solve()
        #
        sol_examples = self.random_pick_solutions(sols, self.N)
        # 
        sol_examples = [np.log(i+self.epsilon_ln) for i in sol_examples]
        return sol_examples

class Back_Linear(Backprop):
    def __init__(self, W, b, N, low_bound=None, low_scope=-1, up_scope=1, epsilon=10e-6):
        super().__init__(low_bound=low_bound, low_scope=low_scope, up_scope=up_scope)
        self.W = W
        self.b = b.reshape(-1, 1)

        self.N = N

        self.epsilon=epsilon

    def _get_appropriate_epsilon(self, W, b):  # get epsilon according to the scale of W and b
        pass   # needs thinking

    def backprop(self, value):
        a = self.W.copy()
        b = self.b.copy()
        sols = LinearEquationSolver(self.W, value.reshape(-1, 1) - self.b, epsilon=self.epsilon).solve()
        k = self.W.dot(sols["special_solution"].reshape(-1, 1))  + self.b
        # print(value-k.flatten())
        # print(sols)
        sol_samples = self.random_pick_solutions(sols, self.N)
        return sol_samples
    
class Back_ReLU(Backprop):
    def __init__(self, N):
        self.N = N

    def backprop(self, value):
        j = value == 0
        l = np.sum(j)
        r = np.random.uniform(-1, 0, (l,))
        value_list = []
        for _ in range(self.N):
            v = value
            v[j] = r
            value_list.append(v)
        return value_list
    
class Back_LeakyReLU(Backprop):
    def __init__(self, slope):
        self.slope = slope

    def backprop(self, value):
        j = value >= 0
        value_list = []
        v = value.copy()
        v[~j] = v[~j]/self.slope
        value_list.append(v)
        return value_list

if __name__ == "__main__":
    from mnist_train import FNN
    from torchvision import datasets, transforms
    import torchvision
    import time
    import os
    import random
    from PIL import Image
    import pickle

    model = FNN()
    model.load_state_dict(torch.load("mnist_fnn.pt"))
    model_params = {}
    for n, p in model.named_parameters():
        model_params[n] = p.data.numpy()

    test_dataset =  datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
    id = random.sample(list(range(len(test_dataset))), 1)[0]

    experiment_id = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    pwd = os.getcwd()
    if not os.path.exists(f"experiments/{experiment_id}"):
        os.mkdir(os.path.join(pwd, "experiments", experiment_id))
    exp_path = f"experiments/{experiment_id}"
    pair = test_dataset[id]
    # pair = test_dataset[10]  ###
    img = pair[0]
    with open(f"{exp_path}/sample_tensor.pkl", "wb") as f:
        pickle.dump(img, f)
    #torchvision.utils.save_image(torch.stack([img]), f"{exp_path}/sample.jpg")
    pil_img = transforms.ToPILImage()(img)
    pil_img.save(f"{exp_path}/sample.jpg")

    # t = transforms.ToTensor()(Image.open(f"{exp_path}/sample.jpg"))

    value = model(img.unsqueeze(0)).squeeze().detach().numpy()
    # print(value)

    mb = Mutiple_Back([
        Back_Softmax(N=1),
        Back_Linear(model_params["layers.2.weight"], model_params["layers.2.bias"], N=1),
        Back_LeakyReLU(slope=0.01),
        Back_Linear(model_params["layers.0.weight"], model_params["layers.0.bias"], N=1, epsilon=10e-12),
    ])

    pic_list = mb.backprop(value)
    pic_list = [torch.tensor(i.astype(np.float64)).view(1, 28, 28) for i in pic_list]

    with open(f"{exp_path}/all_values.pkl", "wb") as f:
        pickle.dump(mb.all_values, f)

    with open(f"{exp_path}/out.pkl", "wb") as f:
        pickle.dump(pic_list[0], f)

    std_pic = (pic_list[0] - torch.min(pic_list[0], dim=2, keepdim=True)[0])/(torch.max(pic_list[0], dim=2, keepdim=True)[0] - torch.min(pic_list[0], dim=2, keepdim=True)[0])
    # pil_img = transforms.ToPILImage()(std_pic)
    # pil_img.save(f"{exp_path}/out.jpg")
    torchvision.utils.save_image(std_pic.repeat(3,1,1), f"{exp_path}/out.jpg")


    def toreadablestr(tensor):
        t = tensor.tolist()
        nt = []
        for i in t:
            ni = []
            for j in i:
                nj = []
                for k in j:
                    nj.append(f"{k:.2f}")
                ni.append(nj)
            nt.append(ni)
        return nt
    import json
    readable_t = {"non_standardized": toreadablestr(pic_list[0]), "standardized": toreadablestr(std_pic)}
    with open(f"{exp_path}/out.txt", "w") as f:
        f.write(json.dumps(readable_t))
