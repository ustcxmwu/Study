#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import pytest
from pathlib import Path
# from pytest_mock import patch
import mock


def name_length(file_name):
    if not Path(file_name).exists():
        raise ValueError("File '{0}' does not exist".format(file_name))
    print(file_name)
    return len(file_name)


@mock.patch.object(Path, 'exists', return_value=True)
def test_name_length0(mocker):
    assert 4 == name_length('test')
    mocker.assert_called_once()

    mocker.return_value = False
    with pytest.raises(ValueError):
        name_length('test')
    assert 2 == mocker.call_count


@mock.patch.object(Path, 'exists', side_effect=TypeError)
def test_name_length1(mocker):
    # mocker.patch.object(Path, 'exists', side_effect=TypeError)
    with pytest.raises(TypeError):
        name_length('test')


@mock.patch.object(Path, 'exists', return_value=True)
@mock.patch('builtins.print', wraps=print)
@mock.patch(__name__ + '.len', wraps=len)
def test_name_length2(mocker, mock_print, mock_len):
    assert 4 == name_length('test')
    assert mock_print.called
    assert mock_len.called


