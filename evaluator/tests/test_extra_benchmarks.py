import json
import math

import numpy as np
from pycocotools import mask as mu

from scoring.protocol_metrics import odinw_metrics, refspatial_accuracy, suite_summary
from run import prompt_for, prediction_record, error_record
from score import primary_metric, referring_accuracy


def test_refspatial_averages_points_then_queries():
    mask = np.zeros((10, 10), dtype=np.uint8, order='F')
    mask[2:5, 2:5] = 1
    row = dict(gt={'target':[mu.encode(mask)]}, status='ok',
               predictions={'target':[[3,3],[8,8]]})
    empty = dict(row, predictions={})
    assert refspatial_accuracy([row, empty])['accuracy'] == .25
    assert refspatial_accuracy([dict(row,predictions={'target':[[-.1,3]]})])['accuracy'] == 0


def test_odinw_ap_keeps_empty_images_and_penalizes_false_positives(tmp_path):
    gt = dict(info={}, categories=[dict(id=1,name='cat')],
              images=[dict(id=i,width=100,height=100) for i in [1,2]],
              annotations=[dict(id=1,image_id=2,category_id=1,bbox=[10,10,20,20],area=400,iscrowd=0)])
    path = tmp_path/'gt.json'
    path.write_text(json.dumps(gt))
    def row(i,preds):
        return dict(status='ok',predictions=preds,evaluation={'coco_gt':str(path),'image_id':i})
    correct = row(2,{'cat':[[10,10,30,30]]})
    assert math.isclose(odinw_metrics([row(1,{}),correct])['mAP'], 1)
    assert math.isclose(
        odinw_metrics([row(1,{'cat':[[0,0,20,20]]}),correct])['mAP'], .5)
    assert odinw_metrics([row(1,{}),row(2,{})])['mAP'] == 0


def test_new_prompt_tasks_and_metric_selection():
    r = dict(dataset_name='ODinW13_Aquarium',task_name='detection',categories=['fish','shark'],
             benchmark_protocol='box')
    assert 'fish, shark' in prompt_for(r,'qwen')
    r.update(dataset_name='RefCOCO+_testA',task_name='referring',categories=['left person'])
    assert 'left person' in prompt_for(r,'rex-omni')
    r.update(benchmark_protocol='point',evaluation={'metric':'refspatial_accuracy','question':'Point to the empty area.'})
    assert prompt_for(r,'elliot-loc') == 'Point to the empty area.'
    assert 'point_2d' in prompt_for(r,'qwen')
    assert primary_metric('point',{'accuracy':.4}) == .4
    assert primary_metric('box',{'mAP':.3}) == .3


def test_metadata_survives_success_and_error():
    r=dict(sample_id='1',dataset_name='ODinW13_Aquarium',image_path='image.jpg',gt={},
           evaluation={'metric':'coco_ap','image_id':1,'coco_gt':'gt.json'})
    adapter=dict(name='test',config_sha256='abc')
    assert prediction_record(r,'prompt',adapter,{},4096)['evaluation'] == r['evaluation']
    assert error_record(r,'prompt',adapter,ValueError('bad'),4096)['evaluation'] == r['evaluation']


def test_referring_is_first_box_accuracy():
    row=dict(status='ok',gt={'target':[[0,0,10,10]]},predictions={'target':[[50,50,60,60],[0,0,10,10]]})
    assert referring_accuracy([row]) == 0


def test_refspatial_suite_weights_questions_not_splits():
    ds={}
    for name,n,acc in [('Location',100,1),('Placement',100,0),('Unseen',50,1)]:
        ds['point/RefSpatialBench_'+name]={'metrics':{'accuracy':acc},'diagnostics':{'samples':n}}
    assert suite_summary(ds)['RefSpatialBench']['score'] == .6
    del ds['point/RefSpatialBench_Unseen']
    assert suite_summary(ds) == {}
