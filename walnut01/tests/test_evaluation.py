import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluation import confusion,scores,boundary_f1

class EvaluationTests(unittest.TestCase):
    def test_exact_small_confusion(self):
        gt=np.array([[0,1,1],[2,2,0]],np.uint8)
        pred=np.array([[0,1,2],[2,0,0]],np.uint8)
        mat=confusion(gt,pred)
        np.testing.assert_array_equal(mat,[[2,0,0],[0,1,1],[1,0,1]])
        s=scores(mat)
        np.testing.assert_allclose(s['iou'],[2/3,1/2,1/3])
        np.testing.assert_allclose(s['dice'],[4/5,2/3,1/2])
    def test_absent_tissue_does_not_inflate_average(self):
        gt=np.zeros((5,5),np.uint8)
        s=scores(confusion(gt,gt))
        self.assertTrue(np.isnan(s['iou'][1:]).all())
        self.assertTrue(np.isnan(s['foreground_miou']))
        pred=gt.copy();pred[1,1]=2
        self.assertEqual(scores(confusion(gt,pred))['iou'][2],0)
    def test_boundary_tolerance(self):
        a=np.zeros((30,30),np.uint8);a[7:18,7:18]=2
        b=np.zeros_like(a);b[7:18,8:19]=2
        self.assertEqual(boundary_f1(a,b,2,1),1)
        self.assertLess(boundary_f1(a,b,2,0),1)
    def test_invalid_classes_rejected(self):
        a=np.zeros((3,3),np.uint8);b=a.copy();b[1,1]=3
        with self.assertRaises(ValueError):confusion(a,b)

if __name__=='__main__':unittest.main()
