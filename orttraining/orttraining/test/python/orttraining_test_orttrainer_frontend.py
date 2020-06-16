import pytest

from onnxruntime.capi.training import orttrainer


def testTrainStepInfo():
    '''Test valid initializations of TrainStepInfo'''

    step_info = orttrainer.TrainStepInfo(all_finite=True, epoch=1, step=2)
    assert step_info.all_finite is True
    assert step_info.epoch == 1
    assert step_info.step == 2

    step_info = orttrainer.TrainStepInfo()
    assert step_info.all_finite is None
    assert step_info.epoch is None
    assert step_info.step is None


@pytest.mark.parametrize("test_input", [
    (-1),
    ('Hello'),
])
def testTrainStepInfoInvalidAllFinite(test_input):
    '''Test invalid initialization of TrainStepInfo'''
    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(all_finite=test_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(epoch=test_input)

    with pytest.raises(AssertionError):
        orttrainer.TrainStepInfo(step=test_input)
