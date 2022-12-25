# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import box_iou_rotated
from ..builder import IOU_CALCULATORS

@IOU_CALCULATORS.register_module()
class RBboxMetrics2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='oc'):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'oc'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        bboxes1 = hbb2obb(bboxes1)
        bboxes2 = hbb2obb(bboxes2)
        
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_metrics(bboxes1.contiguous(), bboxes2.contiguous(), mode,
                              is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

def hbb2obb(bboxes):
    '''
    hbb (torch,tensor) size sush as [m , 4] in (x1,y1,x2,y2)
    obb (torch,tensor) size sush as [m , 5] in (x,y,w,h,theta)
    '''
    xx = torch.div((bboxes[:,0] + bboxes[:,2]),2,rounding_mode='floor')
    yy = torch.div((bboxes[:,1] + bboxes[:,3]),2,rounding_mode='floor')
    ww = abs(bboxes[:,0] - bboxes[:,2])
    hh = abs(bboxes[:,1] - bboxes[:,3])
    tt = torch.zeros_like(hh)
    obb_res = torch.stack((xx,yy,hh,ww,tt),1)
    return obb_res

def rbbox_metrics(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof','gwd','kld','gjsd','center_distance2']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # if mode in ['iou','iof']:
    #     # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    #     clamped_bboxes1 = bboxes1.detach().clone()
    #     clamped_bboxes2 = bboxes2.detach().clone()
    #     clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    #     clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    #     return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)

    if mode == 'gwd':
        g_bboxes1 = xy_wh_r_2_xy_sigma(bboxes1)
        g_bboxes2 = xy_wh_r_2_xy_sigma(bboxes2)
        gwd = get_gwd(g_bboxes1,g_bboxes2)
 

        return gwd

    if mode == 'kld':
        g_bboxes1 = xy_wh_r_2_xy_sigma(bboxes1)
        g_bboxes2 = xy_wh_r_2_xy_sigma(bboxes2)
        kld = get_kld(g_bboxes1,g_bboxes2)

        return kld
        
    if mode == 'bcd':

        return None

    if mode == 'gjsd':
        g_bboxes1 = xy_wh_r_2_xy_sigma(bboxes1)
        g_bboxes2 = xy_wh_r_2_xy_sigma(bboxes2)
        gjsd = get_gjsd(g_bboxes1,g_bboxes2)

        return gjsd

    if mode == 'center_distance2':
        center1 = bboxes1[..., :, None, :2] 
        center2 = bboxes2[..., None, :, :2] 
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6 #

        #distance = torch.sqrt(center_distance2)
    
        return center_distance2
    
    if mode == 'ellipse':
        '''
        inside the ellipse of OBB, assign a positive label
        
        '''
        return None

    


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def get_gwd(pred, target):
    """Gaussian Wasserstein distance loss.
    Derivation and simplification:
        Given any positive-definite symmetrical 2*2 matrix Z:
            :math:`Tr(Z^{1/2}) = λ_1^{1/2} + λ_2^{1/2}`
        where :math:`λ_1` and :math:`λ_2` are the eigen values of Z
        Meanwhile we have:
            :math:`Tr(Z) = λ_1 + λ_2`

            :math:`det(Z) = λ_1 * λ_2`
        Combination with following formula:
            :math:`(λ_1^{1/2}+λ_2^{1/2})^2 = λ_1+λ_2+2 *(λ_1 * λ_2)^{1/2}`
        Yield:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
        For gwd loss the frustrating coupling part is:
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σp^{1/2})^{1/2})`
        Assuming :math:`Z = Σ_p^{1/2} * Σ_t * Σ_p^{1/2}` then:
            :math:`Tr(Z) = Tr(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = Tr(Σ_p^{1/2} * Σ_p^{1/2} * Σ_t)
            = Tr(Σ_p * Σ_t)`
            :math:`det(Z) = det(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = det(Σ_p^{1/2}) * det(Σ_t) * det(Σ_p^{1/2})
            = det(Σ_p * Σ_t)`
        and thus we can rewrite the coupling part as:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σ_p^{1/2})^{1/2})
            = (Tr(Σ_p * Σ_t) + 2 * (det(Σ_p * Σ_t))^{1/2})^{1/2}`

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target
    Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_t = Sigma_t[...,None,:,:2,:2]


    xy_distance = (xy_p[...,:,None,:2] - xy_t[...,None,:,:2]).square().sum(dim=-1)

    whr_distance1 = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance2 = Sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance1 + whr_distance2

    _t_tr = (Sigma_p.matmul(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(0).sqrt()

    
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + whr_distance).clamp(0).sqrt()

    gwd = 1/(1+distance)

    return gwd

def get_kld(pred, target):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    '''
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)

    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    '''
    
    xy_p = xy_p[...,:,None,:2]
    xy_t = xy_t[...,None,:,:2]

    #Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_t = Sigma_t[...,None,:,:2,:2]


    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)
    Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_p_inv = Sigma_p_inv[...,:,None,:2,:2]

    dxy = (xy_p - xy_t).unsqueeze(-1)

    xy_distance = 0.5 * dxy.permute(0, 1, 3, 2).matmul(Sigma_p_inv).matmul(dxy)
    xy_distance=torch.squeeze(xy_distance)
    

    whr_distance = 0.5 * Sigma_p_inv.matmul(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance  + whr_distance)
    distance = distance.clamp(0).sqrt()
    #distance = distance.reshape(_shape[:-1])
    kld = 1/(1+distance)
    #print(kld.size())

    return kld

def get_gjsd(pred, target, alpha=0.5):
    xy_p, Sigma_p = pred  # mu_1, sigma_1
    xy_t, Sigma_t = target # mu_2, sigma_2
    '''
    xy_p = xy_p.half()
    xy_t = xy_t.half()
    Sigma_p = Sigma_p.half()
    Sigma_t = Sigma_t.half()
    
    
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    '''

    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    
    
    xy_p = xy_p[...,:,None,:2]
    xy_t = xy_t[...,None,:,:2]

    # get the inverse of Sigma_p and Sigma_t
    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)
    Sigma_t_inv = torch.stack((Sigma_t[..., 1, 1], -Sigma_t[..., 0, 1],
                               -Sigma_t[..., 1, 0], Sigma_t[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_t_inv = Sigma_t_inv / Sigma_t.det().unsqueeze(-1).unsqueeze(-1)

    Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_p_inv = Sigma_p_inv[...,:,None,:2,:2]
    Sigma_t = Sigma_t[...,None,:,:2,:2]
    Sigma_t_inv = Sigma_t_inv[...,None,:,:2,:2]
    
    Sigma_alpha_ori = ((1-alpha)*Sigma_p_inv + alpha*Sigma_t_inv)

    # get the inverse of Sigma_alpha_ori, namely Sigma_alpha
    Sigma_alpha =  torch.stack((Sigma_alpha_ori[..., 1, 1], -Sigma_alpha_ori[..., 0, 1],
                               -Sigma_alpha_ori[..., 1, 0], Sigma_alpha_ori[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha_ori.size(0), Sigma_alpha_ori.size(1), 2, 2)
    Sigma_alpha = Sigma_alpha / Sigma_alpha_ori.det().unsqueeze(-1).unsqueeze(-1)
    # get the inverse of Sigma_alpha, namely Sigma_alpha_inv
    Sigma_alpha_inv = torch.stack((Sigma_alpha[..., 1, 1], -Sigma_alpha[..., 0, 1],
                               -Sigma_alpha[..., 1, 0], Sigma_alpha[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha.size(0),Sigma_alpha.size(1), 2, 2)
    Sigma_alpha_inv = Sigma_alpha_inv / Sigma_alpha.det().unsqueeze(-1).unsqueeze(-1)

    # mu_alpha
    xy_p = xy_p.unsqueeze(-1)
    xy_t = xy_t.unsqueeze(-1)
    
    mu_alpha_1 = (1-alpha)* Sigma_p_inv.matmul(xy_p) + alpha * Sigma_t_inv.matmul(xy_t)
    mu_alpha = Sigma_alpha.matmul(mu_alpha_1)
     
    
    # the first part of GJSD 
    first_part = (1-alpha) * xy_p.permute(0,1,3,2).matmul(Sigma_p_inv).matmul(xy_p) + alpha * xy_t.permute(0,1,3,2).matmul(Sigma_t_inv).matmul(xy_t) - mu_alpha.permute(0,1,3,2).matmul(Sigma_alpha_inv).matmul(mu_alpha)
    second_part = ((Sigma_p.det() ** (1-alpha))*(Sigma_t.det() ** alpha))/(Sigma_alpha.det())
    second_part = second_part.log()

    if first_part.is_cuda:
        gjsd = 0.5 * (first_part.half().squeeze(-1).squeeze(-1) + second_part.half())
        distance = 1/(1+gjsd)
    else:
        gjsd = 0.5 * (first_part.squeeze(-1).squeeze(-1) + second_part)
        distance = 1/(1+gjsd)

    return distance