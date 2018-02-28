from MKNet import MKNet
from MKSerial import MKSerial
from MKEnv import MKEnv
from MKActionServer import MKActionServer
from MKVideoCapture import MKVideoCapture

def main():
    bundle_frame_size = 8
    camera_number = 1 #usually 0 or 1

    serial_port = '/dev/tty.SLAB_USBtoUART'
    esp32_url = 'ws://192.168.4.1:80/ws'
    mode = 1 # 1 = serial, 0 = websockets

    # game environment init
    num_episodes = 1000000  # Number of episodes to be played
    learning_steps = 100  # Maximum number of learning steps within each episodes
    display_count = 10  # The number of episodes to play before showing statistics.

    # initiate a serial communication
    serial_instance = MKSerial(serial_port)
    
    # Opens communication with either esp32(robot) or serial(emulator)
    action_server_instance = MKActionServer(esp32_url, serial_instance, mode)
    
    #start a videoCapture
    capture_instance = MKVideoCapture(camera_number, bundle_frame_size)

    #start environment
    env_instance = MKEnv(capture_instance, serial_instance, num_episodes, learning_steps, display_count)

    mario_kart = MKNet(serial_instance, action_server_instance, env_instance)
    mario_kart.play()

if __name__ == '__main__':
    main()