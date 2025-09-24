#imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet import resnet18_mnist
from models.mlp import MLP
from distillation_utils import Distiller
from invariances_utils import shift_preserving_shape, validate
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os

temperatures = [1,4,8,16]
random_seeds = [42, 121, 308, 240]

for random_seed in random_seeds:
    os.mkdir("saved_models_undistilled" + "_rs" + str(random_seed))
    os.mkdir("saved_models_undistilled_augmented" + "_rs" + str(random_seed))
    os.mkdir("saved_structurallyinv" + "_rs" + str(random_seed))
    for temperature in temperatures:
        os.mkdir("saved_models_selfdistill_unshifted" + "_rs" + str(random_seed) + "_temp" + str(temperature))
        os.mkdir("saved_models_selfdistill_augmented" + "_rs" + str(random_seed) + "_temp" + str(temperature))
        os.mkdir("saved_models_distill_shiftinv1_to_mlp1" + "_rs" + str(random_seed) + "_temp" + str(temperature))
        os.mkdir("saved_models_distill_shiftinv2_to_mlp2" + "_rs" + str(random_seed) + "_temp" + str(temperature))

in_channels = 1
num_epochs = 10
num_classes = 10
batch_size = 64
lr = 0.001
TRAIN = True
device = 'cuda'

#results excel
res = { "Model": [],
       "temp" : [],
       "accuracy" : [],
       "NLL" : [],
       "ECEL" : [],
       "SINV" : [],
       "T1agree": [],
        "KLDiv": [],
}

# Define a custom dataset that combines MNIST and additional data
class ShiftAugmentedMNIST(Dataset):
    def __init__(self, mnist_dataset, translation_times : int = 5, max_shift : int = 5):
        self.mnist_dataset = mnist_dataset
        directions = ["u","d","l","r"]
        self.translations = []
        for i in range(len(self.mnist_dataset)):
            img, label = self.mnist_dataset[i]
            img = img.squeeze()
            for t in range(translation_times):
                sh = shift_preserving_shape(img, direction=directions[np.random.randint(0,4)],
                                            max_shift=max_shift).unsqueeze(0)
                if sh is not None:
                    self.translations.append((sh, label))

    def __getitem__(self, index):
        if index < len(self.mnist_dataset):
            return self.mnist_dataset[index]
        else:
            return self.translations[index - len(self.mnist_dataset)]

    def __len__(self):
        return len(self.mnist_dataset) + len(self.translations)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_augmented_dataset = ShiftAugmentedMNIST(train_dataset)
train_augmented_loader = DataLoader(dataset=train_augmented_dataset, batch_size=batch_size, shuffle=True)

try:
    for random_seed in random_seeds:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

        print(f"RANDOM SEED {random_seed}\n\n")
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_cached())

        #Loading undistilled MLP
        if TRAIN:
            undistilled_mlp = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                hidden_layers= 4, device='cuda')
            criterion_mlp = torch.nn.CrossEntropyLoss()
            optimizer_mlp = torch.optim.Adam(undistilled_mlp.parameters(), lr=lr)
            undistilled_mlp.training_loop(train_loader=train_loader, optimizer=optimizer_mlp, criterion=criterion_mlp, 
                    num_epochs=5, save_path_folder="saved_models_undistilled" + "_rs" + str(random_seed))
        if not TRAIN:
            undistilled_mlp = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                    hidden_layers= 4, device='cuda', from_saved_state_dict="saved_models_undistilled" + "_rs" + str(random_seed) + "/mlp")
        print("MLP WITHOUT SHIFT AUGMENTED DATA")
        valid = validate(model=undistilled_mlp, weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
        valid["Model"] = "undistilled_unaugmented_mlp"
        for key in res:
            if key in valid:
                res[key].append(valid[key])
            else:
                res[key].append("na")

        #Loading undistilled MLP augmented
        if TRAIN:
            undistilled_mlp_augmented = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                hidden_layers= 4, device='cuda')
            criterion_mlp = torch.nn.CrossEntropyLoss()
            optimizer_mlp = torch.optim.Adam(undistilled_mlp_augmented.parameters(), lr=lr)
            undistilled_mlp_augmented.training_loop(train_loader=train_augmented_loader, optimizer=optimizer_mlp, criterion=criterion_mlp, 
                    num_epochs=5, save_path_folder="saved_models_undistilled_augmented"  + "_rs" + str(random_seed))
        if not TRAIN:
            undistilled_mlp_augmented = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                    hidden_layers= 4, device='cuda', from_saved_state_dict="saved_models_undistilled_augmented" + "_rs" + str(random_seed) + "/mlp")
        print("MLP TRAINED WITH SHIFT AUGMENTED DATA")
        valid = validate(model=undistilled_mlp_augmented, weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
        valid["Model"] = "undistilled_augmented_mlp"
        for key in res:
            if key in valid:
                res[key].append(valid[key])
            else:
                res[key].append("na")

        #Obtaining ResNet
        resnet_path = "saved_structurallyinv" + "_rs" + str(random_seed) + "/model_1"
        resnet = resnet18_mnist().to(device)
        if TRAIN:
            criterion_resnet = torch.nn.CrossEntropyLoss()
            optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=lr)
            # model training
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    outputs = resnet(images.to('cuda'))
                    loss = criterion_resnet(outputs, labels.to('cuda'))

                    optimizer_resnet.zero_grad()
                    loss.backward()
                    optimizer_resnet.step()

                    if (i + 1) % 100 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # Save the trained model
            torch.save(resnet.state_dict(), resnet_path)
            print(f"Model saved as {resnet_path}!")
        if not TRAIN:
            state_dict = torch.load(resnet_path)
            resnet.load_state_dict(state_dict=state_dict)
        print("STRUCTURALLY SHIFT INVARIANT MODEL 1")
        valid = validate(model=resnet, weights_file=None, valid_data=test_loader, device=device, is_mlp= False)
        valid["Model"] = "shift_invariant_arch_1"
        for key in res:
            if key in valid:
                res[key].append(valid[key])
            else:
                res[key].append("na")

        #Obtaining ResNet'
        resnet_prime_path = "saved_structurallyinv" + "_rs" + str(random_seed) + "/model_2"
        resnet_prime = resnet18_mnist().to(device)
        if TRAIN:
            criterion_resnet_prime = torch.nn.CrossEntropyLoss()
            optimizer_resnet_prime = torch.optim.Adam(resnet_prime.parameters(), lr=lr)
            # model training
            for epoch in range(2):
                for i, (images, labels) in enumerate(train_loader):
                    outputs = resnet_prime(images.to('cuda'))
                    loss = criterion_resnet_prime(outputs, labels.to('cuda'))

                    optimizer_resnet_prime.zero_grad()
                    loss.backward()
                    optimizer_resnet_prime.step()

                    if (i + 1) % 100 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # Save the trained model
            torch.save(resnet_prime.state_dict(), resnet_prime_path)
            print(f"Model saved as {resnet_prime_path}!")
        if not TRAIN:
            state_dict = torch.load(resnet_prime_path)
            resnet_prime.load_state_dict(state_dict=state_dict)
        print("STRUCTURALLY SHIFT INVARIANT MODEL 2")
        valid = validate(model=resnet_prime, weights_file=None, valid_data=test_loader, device=device, is_mlp= False)
        valid["Model"] = "shift_invariant_arch_2"
        for key in res:
            if key in valid:
                res[key].append(valid[key])
            else:
                res[key].append("na")
            
        for temperature in temperatures:
            print(f"TEMPERATURE{temperature}\n\n")

            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated())
            print(torch.cuda.memory_cached())

            #Self distilling MLP (only from unshifted data)
            save_path_folder = "saved_models_selfdistill_unshifted" + "_rs" + str(random_seed) + "_temp" + str(temperature) + "/"
            teacher_path = "saved_models_undistilled" + "_rs" + str(random_seed) + "/mlp"
            mlp_student = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device)
            mlp_teacher = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device, from_saved_state_dict=teacher_path)
            print("MODEL MLP TEACHER NOT SHIFT INVARIANT")
            validate(model=mlp_teacher, weights_file=None, valid_data=test_loader, device=device, is_mlp= True)

            if TRAIN:
                selfdistiller = Distiller(student=mlp_student, teacher=mlp_teacher, device=device, temp=temperature, lr=0.001, is_teacher_mlp=True)
                selfdistiller.distill(train_data=train_loader, valid_data=test_loader, save_path_folder= save_path_folder) # TODO the model will not be saved in distill() or is it inherited?? put code for saving student
            if not TRAIN:
                print("Loading params")
                selfdistiller = Distiller(student=mlp_student, teacher=mlp_teacher, device=device, temp=temperature, lr=0.001,
                                    load_student_from_path = save_path_folder + 'distiller', is_teacher_mlp=True)

            print("MODEL MLP STUDENT FROM MLP TEACHER NOT SHIFT INVARIANT")
            valid = validate(model=selfdistiller.get_student(), weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
            valid["Model"] = "selfdistill_teacher_unshifted"
            valid["temp"] = selfdistiller.get_temperature()
            fid = selfdistiller.compute_fidelity(test_loader)
            for key in fid:
                valid[key] = fid[key]
            for key in res:
                if key in valid:
                    res[key].append(valid[key])
                else:
                    res[key].append("na")


            #Self distilling MLP (augmented)
            save_path_folder = "saved_models_selfdistill_augmented" + "_rs" + str(random_seed) + "_temp" + str(temperature) + "/"
            teacher_path = "saved_models_undistilled_augmented" + "_rs" + str(random_seed) + "/mlp"
            mlp_student = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device)

            mlp_teacher = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device, from_saved_state_dict=teacher_path)
            print("MODEL MLP TEACHER SHIFT INVARIANT")
            validate(model=mlp_teacher, weights_file=teacher_path, valid_data=test_loader, device=device, is_mlp= True)

            if TRAIN:
                selfdistiller = Distiller(student=mlp_student, teacher=mlp_teacher, device=device, temp=temperature, lr=0.001, is_teacher_mlp=True)
                selfdistiller.distill(train_data=train_loader, valid_data=test_loader, save_path_folder= save_path_folder)
            if not TRAIN:
                print("Loading params")
                selfdistiller = Distiller(student=mlp_student, teacher=mlp_teacher, device=device, temp=temperature, lr=0.001,
                                    load_student_from_path = save_path_folder + 'distiller', is_teacher_mlp=True)
            print("MODEL MLP STUDENT FROM MLP TEACHER SHIFT INVARIANT")
            valid = validate(model=mlp_student, weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
            valid["Model"] = "selfdistill_teacher_shifted"
            valid["temp"] = selfdistiller.get_temperature()
            fid = selfdistiller.compute_fidelity(test_loader)
            for key in fid:
                valid[key] = fid[key]
            for key in res:
                if key in valid:
                    res[key].append(valid[key])
                else:
                    res[key].append("na")

            #Distilling RESNET to MLP
            save_path_folder_1 = "saved_models_distill_shiftinv1_to_mlp1" + "_rs" + str(random_seed) + "_temp" + str(temperature) + "/"
            save_path_folder_2 = "saved_models_distill_shiftinv2_to_mlp2" + "_rs" + str(random_seed) + "_temp" + str(temperature) + "/"
            teacher_path_1 = "saved_structurallyinv" + "_rs" + str(random_seed) + "/model_1"
            teacher_path_2 = "saved_structurallyinv" + "_rs" + str(random_seed) + "/model_2"

            mlp_student_1 = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device)

            mlp_student_2 = MLP(input_dim = 784, output_dim= num_classes, hidden_size= 256,
                        hidden_layers= 4, device=device)

            teacher_1 = resnet18_mnist().to(device)
            state_dict = torch.load(teacher_path_1)
            teacher_1.load_state_dict(state_dict=state_dict)
            print("STRUCTURALLY INVARIANT MODEL 1")
            validate(model=teacher_1, weights_file=None, valid_data=test_loader, device=device, is_mlp= False)

            #TODO: use different architecture than mlp_teacher_1 (at the moment it is the same)
            teacher_2 = resnet18_mnist().to(device)
            state_dict = torch.load(teacher_path_2)
            teacher_2.load_state_dict(state_dict=state_dict)
            print("STRUCTURALLY INVARIANT MODEL 2")
            validate(model=teacher_2, weights_file=None, valid_data=test_loader, device=device, is_mlp= False)

            if TRAIN:
                student1_teacher1 = Distiller(student=mlp_student_1, teacher=teacher_1, device=device, temp=temperature, lr=0.001)
                student1_teacher1.distill(train_data=train_loader, valid_data=test_loader, save_path_folder= save_path_folder_1) # TODO the model is not saved in distill() or is it inherited??

                student2_teacher2 = Distiller(student=mlp_student_2, teacher=teacher_2, device=device, temp=temperature, lr=0.001)
                student2_teacher2.distill(train_data=train_loader, valid_data=test_loader, save_path_folder= save_path_folder_2) # TODO the model is not saved in distill() or is it inherited??
            if not TRAIN:
                student1_teacher1 = Distiller(student=mlp_student_1, teacher=teacher_1, device=device, temp=temperature, lr=0.001,
                                    load_student_from_path = save_path_folder_1 + 'distiller')

                student2_teacher2 = Distiller(student=mlp_student_2, teacher=teacher_2, device=device, temp=temperature, lr=0.001,
                                    load_student_from_path = save_path_folder_2 + 'distiller')

            print("STUDENT TEACHER STRUCTURALLY INVARIANT MODEL 1")
            valid = validate(model=student1_teacher1.get_student(), weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
            valid["Model"] = "distill_teacher_structurally_invariant_model_1"
            valid["temp"] = student1_teacher1.get_temperature()
            fid = student1_teacher1.compute_fidelity(test_loader)
            for key in fid:
                valid[key] = fid[key]
            for key in res:
                if key in valid:
                    res[key].append(valid[key])
                else:
                    res[key].append("na")

            print("STUDENT TEACHER STRUCTURALLY INVARIANT MODEL 2")
            valid = validate(model=student2_teacher2.get_student(), weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
            valid["Model"] = "distill_teacher_structurally_invariant_model_2"
            valid["temp"] = student2_teacher2.get_temperature()
            fid = student2_teacher2.compute_fidelity(test_loader)
            for key in fid:
                valid[key] = fid[key]
            for key in res:
                if key in valid:
                    res[key].append(valid[key])
                else:
                    res[key].append("na")

            print("STUDENT TEACHER STRUCTURALLY INVARIANT MODEL 2 ON TEACHER 1")
            student2_teacher1 = Distiller(student=student2_teacher2.get_student(), teacher=teacher_1, temp=temperature, device=device, lr=0.001)
            valid = validate(model=student2_teacher1.get_student(), weights_file=None, valid_data=test_loader, device=device, is_mlp= True)
            valid["Model"] = "student_trained_on_2_teacher_1"
            valid["temp"] = student2_teacher1.get_temperature()
            fid= student2_teacher1.compute_fidelity(test_loader)
            for key in fid:
                valid[key] = fid[key]
            for key in res:
                if key in valid:
                    res[key].append(valid[key])
                else:
                    res[key].append("na")


    res = pd.DataFrame(res)
    res.to_excel('solutions.xlsx', index=False)

    columns = ["accuracy", "NLL", "ECEL", "SINV", "T1agree", "KLDiv"]
    for col in columns:
        if col in ["T1agree", "KLDiv"]:
            res[col] = res[col].transform(lambda x: x if x != "na" else None)
        mean = res.groupby(["Model","temp"])[col].transform("mean")
        sensitivity = res.groupby(["Model", "temp"])[col].transform(lambda x: x.max()- x.min())
        res[col] = mean
        res[col + " sensitivity"] = sensitivity
    res = res.drop_duplicates(subset=['Model', 'temp'])
    res = res.reset_index(drop=True)
    cols = ["T1agree", "KLDiv", "T1agree sensitivity", "KLDiv sensitivity"]
    for col in cols:
        res[col] = res[col].fillna("na")
    res.to_excel("solutions_processed.xlsx")

    print("done")
except KeyboardInterrupt:
    res = pd.DataFrame(res)
    res.to_excel('solutions_backup.xlsx', index=False)

    print("done")
finally:
    res = pd.DataFrame(res)
    res.to_excel('solutions_backup.xlsx', index=False)

    print("done")