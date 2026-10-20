"""Evaluation only. Inference executable never imports this module or labels."""
import numpy as np
import cv2

def confusion(reference, prediction, valid=None):
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError('Labelmaps must be matching 2D arrays')
    if not np.isin(reference,[0,1,2]).all() or not np.isin(prediction,[0,1,2]).all():
        raise ValueError('Unexpected class identifier')
    if valid is None:
        valid=np.ones(reference.shape,dtype=bool)
    return np.bincount((reference[valid].astype(int)*3+prediction[valid]).ravel(),minlength=9).reshape(3,3)

def scores(matrix):
    tp=np.diag(matrix).astype(float)
    true=matrix.sum(axis=1)
    pred=matrix.sum(axis=0)
    union=true+pred-tp
    def divide(a,b):
        return np.divide(a,b,out=np.full(3,np.nan),where=b!=0)
    iou=divide(tp,union)
    dice=divide(2*tp,true+pred)
    return {'iou':iou,'dice':dice,'precision':divide(tp,pred),
            'recall':divide(tp,true),'miou':float(np.nanmean(iou)),
            'foreground_miou':float(np.nanmean(iou[1:])) if np.any(np.isfinite(iou[1:])) else np.nan,
            'accuracy':float(tp.sum()/matrix.sum()) if matrix.sum() else np.nan}

def boundary_f1(reference,prediction,cls,tolerance=1):
    kernel=np.ones((3,3),np.uint8)
    def boundary(mask):
        u=mask.astype('uint8')
        return (u-cv2.erode(u,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=1)).astype(bool)
    a=boundary(reference==cls);b=boundary(prediction==cls)
    if not a.any() and not b.any():
        return np.nan
    if not a.any() or not b.any():
        return 0.0
    near_a=cv2.dilate(a.astype('uint8'),kernel,iterations=tolerance).astype(bool)
    near_b=cv2.dilate(b.astype('uint8'),kernel,iterations=tolerance).astype(bool)
    precision=np.sum(b&near_a)/np.sum(b)
    recall=np.sum(a&near_b)/np.sum(a)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0

def json_finite(value):
    if isinstance(value,np.ndarray):return [json_finite(v) for v in value.tolist()]
    if isinstance(value,dict):return {k:json_finite(v) for k,v in value.items()}
    if isinstance(value,list):return [json_finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value
