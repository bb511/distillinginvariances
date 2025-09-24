import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchmetrics

def visualize_tensor_as_image(tensor):
    """
    Visualizes a 2D square matrix (tensor) as an image.

    Parameters:
    - tensor (numpy.ndarray): The input square matrix to be visualized as an image.

    Raises:
    - ValueError: If the input tensor is not a square matrix.

    Returns:
    - None: Displays the image using Matplotlib.
    """
    if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input tensor must be a square matrix.")

    plt.imshow(tensor, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

def shift_not_preserving_shape(image, direction : str, max_shift: int, debug: bool = False):
    """
    Apply a non-preserving shape shift to a 2D tensor/image.

    Parameters:
    - image (torch.Tensor): Input 2D tensor or image.
    - direction (str): Direction of the shift ("u" for up, "d" for down, "l" for left, "r" for right).
    - max_shift (int): Maximum number of pixels to shift.
    - debug (bool, optional): If True, visualize the image before and after the shift for debugging purposes. Default is False.

    Returns:
    - torch.Tensor: The shifted 2D tensor or image.

    Raises:
    - ValueError: If an incorrect direction value is passed.
    """
    img = torch.clone(image)
    shift = np.random.randint(low=1, high= max_shift+1)
    if debug:
        visualize_tensor_as_image(img)
    if direction == "u":
        img = torch.roll(img, -shift, 0)
        img[-shift:,:] = torch.full(img[-shift:,:].shape, -1)
    elif direction == "d":
        img = torch.roll(img, shift, 0)
        img[:shift,:] = torch.full(img[:shift,:].shape, -1)
    elif direction == "l":
        img = torch.roll(img, -shift, 1)
        img[:,-shift:] = torch.full(img[:,-shift:].shape, -1)
    elif direction == "r":
        img = torch.roll(img, shift, 1)
        img[:,:shift] = torch.full(img[:,:shift].shape, -1)
    else:
        raise ValueError("wrong value passed")
    if debug:
        visualize_tensor_as_image(img)
    return img

def shift_preserving_shape(image, direction : str, max_shift: int, debug: bool = False):
    """
    Apply a preserving shape shift to a 2D tensor/image. 
    This function does not apply a shift of some amount in 
    some direction if that would result in losing some pixels of the digit. 
    Instead, it tries to shift the image in all other directions with a maximum shift given
    by max_shift. 

    Parameters:
    - image (torch.Tensor): Input 2D tensor or image.
    - direction (str): Preferred direction of the shift ("u" for up, "d" for down, "l" for left, "r" for right).     
    This function does not apply a shift in 
    the specified direction if that would result in losing some pixels of the digit. 
    Instead, it tries to shift the image in all other directions with a maximum shift given
    by max_shift. 
    - max_shift (int): Maximum number of pixels to shift.
    - debug (bool, optional): If True, visualize the image before and after the shift for debugging purposes. Default is False.

    Returns:
    - torch.Tensor: The shifted 2D tensor or image. If the shift is not possible in any direction for any amount, None is returned.

    Raises:
    - ValueError: If an incorrect direction value is passed.
    """
    initial_dir = direction
    img = torch.clone(image)
    shift = np.random.randint(low=1, high= max_shift+1)
    shift = max_shift
    if debug:
        visualize_tensor_as_image(img)
    row_length = img.shape[1]
    col_length = img.shape[0]
    if direction == "u":
        while shift > 0 and torch.sum(img[:shift, :]) != -1 * shift * col_length:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "d":
                print("Image could not be shifted.")
                return None
            direction = "d"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, -shift, 0)
    elif direction == "d":
        while shift > 0 and torch.sum(img[-shift:, :]) != -1 * shift * col_length:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "l":
                print("Image could not be shifted.")
                return None
            direction = "l"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, shift, 0)
    elif direction == "l":
        while shift > 0 and torch.sum(img[:, :shift]) != -1 * row_length * shift:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "r":
                print("Image could not be shifted")
                return None
            direction = "r"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, -shift, 1)
    elif direction == "r":
        while shift > 0 and torch.sum(img[:, -shift:]) != -1 * row_length * shift:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "u":
                print("Image could not be shifted")
                return None
            direction = "u"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, shift, 1)
    else:
        raise ValueError("wrong value passed")
    if debug:
        visualize_tensor_as_image(img)
    return img

def invariance_measure(labels_normal, labels_shifted):
    """
    Calculate the IM between two sets of probability distributions.

    Parameters:
    - labels_normal (torch.Tensor): Logits (before softmax) for the normal input (before transformation).
    - labels_shifted (torch.Tensor): Logits (before softmax) for the transformed input.

    Returns:
    - torch.Tensor: Mean L2 norm of the differences between the probability distributions.

    Note:
    The input tensors are assumed to represent batched probability distributions over classes.
    The softmax function is applied to ensure that the inputs are normalized into probability distributions.
    """
    labels_normal = torch.softmax(labels_normal, dim=1) #batch classes
    labels_shifted = torch.softmax(labels_shifted, dim=1) #batch classes
    return torch.mean(torch.norm(labels_normal - labels_shifted, dim=1))

def test_IM(loader, model, model2, device, debug: bool = False):
    """
    Evaluate a model's invariance using shifted and non-shifted images.

    Parameters:
    - loader (torch.utils.data.DataLoader): Data loader for the dataset.
    - model (torch.nn.Module): The primary model to evaluate.
    - model2 (torch.nn.Module): Second model for comparison.
    - device (torch.device): The device on which the evaluation should be performed.
    - debug (bool, optional): If True, visualize images for debugging purposes. Default is False.

    Returns:
    - torch.Tensor: Mean invariance measure (IM) over the dataset.
    """
    model2.eval()
    model.eval()
    directions = ["u", "d", "l", "r"]
    invariance_measures = []
    invariance_measure_model2 = []
    n = 0
    correct_normal = 0
    correct_shifted = 0
    correct_normal_model2 = 0
    correct_shifted_model2 = 0
    random_affine = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))

    for images,labels in loader:
        #images: batch channel rows cols
        labels = labels.to(device)
        images = images.squeeze().to(device)
        shifted = []
        non_shifted = []
        for img in images:
            np.random.shuffle(directions)
            sh = shift_preserving_shape(img, direction=directions[0], max_shift=5)
            if sh is not None:
                n = n + 1
                if debug:
                    visualize_tensor_as_image(img.cpu())
                    visualize_tensor_as_image(sh.cpu())
                shifted.append(sh.unsqueeze(0))
                non_shifted.append(img.unsqueeze(0))
        shifted = torch.cat(shifted, dim=0)
        non_shifted = torch.cat(non_shifted, dim=0)
        with torch.no_grad():
            model2_unsh = model2(non_shifted.unsqueeze(1))
            model2_sh = model2(shifted.unsqueeze(1))
            shifted = shifted.view(-1, shifted.shape[-1] * shifted.shape[-2])
            non_shifted = non_shifted.view(-1, non_shifted.shape[-1] * non_shifted.shape[-2])
            unshifted_labels = model(non_shifted)
            shifted_labels = model(shifted)
        correct_normal = correct_normal + torch.sum(torch.max(unshifted_labels, dim = 1)[1] == labels).item()
        correct_shifted = correct_shifted + torch.sum(torch.max(shifted_labels, dim= 1)[1] == labels).item()
        correct_normal_model2 = correct_normal_model2 + torch.sum(torch.max(model2_unsh, dim = 1)[1] == labels).item()
        correct_shifted_model2 = correct_shifted_model2 + torch.sum(torch.max(model2_sh, dim= 1)[1] == labels).item()
        invariance_measure_model2.append(invariance_measure(model2_unsh, model2_sh).unsqueeze(0))
        invariance_measures.append(invariance_measure(unshifted_labels, shifted_labels).unsqueeze(0))
    print(f"Correct normal: {correct_normal/n}\n"
          + f"Correct shifted: {correct_shifted/n}\n"
          + f"Correct model2 normal: {correct_normal_model2/n}\n"
          + f"Correct model2 shifted: {correct_shifted_model2/n}\n")
    
    plt.subplot(1,2,1)
    shifted_image = transforms.ToPILImage()(sh)
    plt.imshow(shifted_image, cmap="gray")
    plt.subplot(1,2,2)
    shifted_image = transforms.ToPILImage()(img)
    plt.imshow(shifted_image, cmap="gray")
    print(torch.mean(torch.cat(invariance_measure_model2)))
    return torch.mean(torch.cat(invariance_measures))

def test_IM_single(loader, model, device, is_mlp, debug: bool = False):
    """
    Evaluate a model's invariance using shifted and non-shifted images for a single model.

    Parameters:
    - loader (torch.utils.data.DataLoader): Data loader for the dataset.
    - model (torch.nn.Module): The model to evaluate.
    - device (torch.device): The device on which the evaluation should be performed.
    - is_mlp (bool): set to True if the model is an MLP.
    - debug (bool, optional): If True, visualize images for debugging purposes. Default is False.

    Returns:
    - torch.Tensor: Mean invariance measure over the dataset.
    
    """
    model.eval()
    directions = ["u", "d", "l", "r"]
    invariance_measures = []
    n = 0
    correct_normal = 0
    correct_shifted = 0

    for images,labels in loader:
        #images: batch channel rows cols
        labels = labels.to(device)
        images = images.squeeze().to(device)
        shifted = []
        non_shifted = []
        for img in images:
            np.random.shuffle(directions)
            sh = shift_preserving_shape(img, direction=directions[0], max_shift=5)
            if sh is not None:
                n = n + 1
                if debug:
                    visualize_tensor_as_image(img.cpu())
                    visualize_tensor_as_image(sh.cpu())
                shifted.append(sh.unsqueeze(0))
                non_shifted.append(img.unsqueeze(0))
        shifted = torch.cat(shifted, dim=0)
        non_shifted = torch.cat(non_shifted, dim=0)
        with torch.no_grad():
            if is_mlp:
                shifted = shifted.view(-1, shifted.shape[-1] * shifted.shape[-2])
                non_shifted = non_shifted.view(-1, non_shifted.shape[-1] * non_shifted.shape[-2])
                unshifted_labels = model(non_shifted)
                shifted_labels = model(shifted)
            else:
                unshifted_labels = model(non_shifted.unsqueeze(1))
                shifted_labels = model(shifted.unsqueeze(1))
            
        correct_normal = correct_normal + torch.sum(torch.max(unshifted_labels, dim = 1)[1] == labels).item()
        correct_shifted = correct_shifted + torch.sum(torch.max(shifted_labels, dim= 1)[1] == labels).item()
        invariance_measures.append(invariance_measure(unshifted_labels, shifted_labels).unsqueeze(0))
    print(f"Correct normal: {correct_normal/n}\n"
          + f"Correct shifted: {correct_shifted/n}\n")
    
    return torch.mean(torch.cat(invariance_measures))

def validate(model: torch.nn.Module, weights_file: str, valid_data: DataLoader, device: str, is_mlp: bool):
    """
    Validate a model's performance on a validation dataset.

    Parameters:
    - model (torch.nn.Module): The model to be validated.
    - weights_file (str): Path to the file containing the model weights. If None, no weights are loaded.
    - valid_data (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - device (str): Device on which to perform the validation (e.g., "cuda" or "cpu").
    - is_mlp (bool): set to True if the model is an MLP.

    Returns:
    - dict: A dictionary containing validation metrics (accuracy, negative log likelihood, ECE, and shift invariance).
    """
    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file))
    model.to(device)
    nll = torch.nn.NLLLoss()
    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=10)

    batch_accu_sum = 0
    batch_nlll_sum = 0
    batch_ecel_sum = 0
    totnum_batches = 0
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(valid_data):
            x = x.to(device)
            y_true = y.to(device)
            if (is_mlp):
                y_pred = model(x.view(-1,784))
            else:
                y_pred = model(x)
            accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)

            log_probs = torch.nn.LogSoftmax(dim=1)(y_pred)
            nll_loss = nll(log_probs, y_true)
            ece_loss = ece(y_pred, y_true)

            batch_accu_sum += accu
            batch_nlll_sum += nll_loss
            batch_ecel_sum += ece_loss
            totnum_batches += 1

        metrics = {
            "accuracy": (batch_accu_sum / totnum_batches).cpu().item(),
            "NLL": (batch_nlll_sum / totnum_batches).cpu().item(),
            "ECEL": (batch_ecel_sum / totnum_batches).cpu().item(),
            "SINV": test_IM_single(valid_data, model, device, is_mlp).cpu().item()
        }
        for key, value in metrics.items():
            print(f"{key}: {value:.8f}")
            print("")

        return metrics