#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import pytest
from unittest import mock


@pytest.fixture(scope='function', params=[1, 2, 3])
def mock_data_params(request):
    return request.param


def test_not_2(mock_data_params):
    print('test_data: {}'.format(mock_data_params))
    assert mock_data_params != 2


def _call(x):
    return '{} _call'.format(x)


def func(x):
    return '{} func'.format(_call(x))

def test_func_normal():
    for x in [11, 22, 33]:
        assert func(x) == '{} _call func'.format(x)


@mock.patch('pytest_client._call', return_value='None')
def test_func_mock(mock_call):
    for x in [11, 22, 33]:
        ret = func(x)
        assert mock_call.called
        assert mock_call.call_args == ((x, ),)
        assert ret == 'None func'



