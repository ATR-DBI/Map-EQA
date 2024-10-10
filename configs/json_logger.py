import logging
import datetime
from pytz import timezone
from pythonjsonlogger import jsonlogger


# https://github.com/madzak/python-json-logger#customizing-fields
class JsonFormatter(jsonlogger.JsonFormatter):

    def parse(self):
        """
        他に出したいフィールドがあったらこのリストに足す
        https://docs.python.jp/3/library/logging.html
        """
        return [
            'infos',
            'timestamp',
            'level',
            'scene',
        ]

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # https://qiita.com/yoppe/items/4260cf4ddde69287a632
            now = datetime.datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%dT%H:%M:%S%z')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


def getLogger(module_name):
    """プロジェクトごとにハンドラの設定などをしたい場合はここでやる"""
    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
