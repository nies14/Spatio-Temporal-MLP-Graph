import torch
import numpy as np



def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))



def test_calculation(predicted, target, action, error_sum, data_type, show_protocol2=False):
    """
    get test error sum accumulation
    :param predicted:
    :param target:
    :param action:
    :param action_error_sum:
    :param eva_by_subj:
    :return:
    """
    if data_type == 'h36m':
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        if show_protocol2:
            error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    elif data_type.startswith('STB'):
        error_sum = mjmpe_directly(predicted, target, error_sum)

    return error_sum

def mjmpe_directly(predicted, target,action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    # action_error_sum[0] += num
    # action_error_sum[1] += num * torch.mean(dist).item()
    action_error_sum.update(torch.mean(dist).item() * num, num)
    return action_error_sum


def mjmpe_by_action_subject(predicted, target,action,action_error_sum,subject):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    if len(set(list(action))) == 1 and len(set(list(subject))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name][subject][0] += num
        action_error_sum[action_name][subject][1] += num*torch.mean(dist).item()
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name][subject][0] += 1
            action_error_sum[action_name][subject][1] += dist[i].item()
    return action_error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    # some action name have space, example: walking 1, change it to walking
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
    return action_error_sum

def mpjpe_by_action_p2(predicted, target,action,action_error_sum):
    """
    Aligned Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred,gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        # action_error_sum[action_name][2] += num*np.mean(dist)
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            # action_error_sum[action_name][2] += dist[i].item()
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
    return action_error_sum


#
def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),axis=len(target.shape) - 2)




def sym_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    if dataset == 'h36m':
        if keypoints.startswith('sh') or keypoints.startswith('hr') or keypoints.startswith('gt'):
            left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
            right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
        else:
            left_bone = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
            right_bone = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]
        for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
            left_part = pred_out[:,:,i_left]-pred_out[:,:,j_left]
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.norm(left_part, dim=- 1) - torch.norm(right_part, dim=- 1))
    elif dataset.startswith('STB'):
        loss_sym = 0
    return loss_sym

def bone_length_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the consistency of the bone length in sequences
    :return:
    """
    loss_bone = 0
    if pred_out.size(1) == 1:
        return 0
    if dataset == 'h36m' and keypoints != 'sh_ft_h36m':
        bone_id = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13),(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)
            ,(0,7),(7,8),(8,9),(9,10)]
        for (i,j) in bone_id:
            bone = torch.norm(pred_out[:,:,i]-pred_out[:,:,j],dim=-1)#N*T
            loss_bone += torch.sum(torch.var(bone,dim=1))#N

    return loss_bone






