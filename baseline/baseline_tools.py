from datetime import datetime


def logging(info: str):
    print('\n\r' + '[INFO]' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          '\n\r' + str(info))


def get_time() -> str:
    return str(datetime.now().strftime("%m-%d-%H-%M"))