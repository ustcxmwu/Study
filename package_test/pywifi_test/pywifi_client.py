import pywifi
import time
from pywifi import const


def main():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    print(iface)


if __name__ == "__main__":
    main()