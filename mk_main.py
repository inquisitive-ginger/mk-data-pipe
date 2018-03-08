import mxnet as mx

from MKNet import MKNet
from MKSerial import MKSerial
from MKEnv import MKEnv
from MKActionServer import MKActionServer
from MKVideoCapture import MKVideoCapture
from MKResults import MKResults
from MKDirectionDetect import MKDirectionDetect

def main():
    # video capture parameters
    bundle_frame_size = 2
    camera_number = 1 #usually 0 or 1

    # communication parameters
    serial_port = '/dev/tty.SLAB_USBtoUART'
    esp32_url = 'ws://192.168.4.1:80/ws'
    mode = 1 # 1 = serial, 0 = websockets

    # Loading model to predict direction of Mario
    directionModel = MKDirectionDetect()
    directionModel.dirNet.load_params("./model.params", ctx=mx.cpu())

    # game environment init
    num_episodes = 1000000  # Number of episodes to be played
    learning_steps = 15  # Maximum number of learning steps within each episodes
    display_count = 10  # The number of episodes to play before showing statistics.

    # initiate a serial communication
    serial_instance = MKSerial(serial_port)
    
    # Opens communication with either esp32(robot) or serial(emulator)
    action_server_instance = MKActionServer(esp32_url, serial_instance, mode)
    
    # start a videoCapture
    capture_instance = MKVideoCapture(camera_number, bundle_frame_size)

    # start environment
    env_instance = MKEnv(capture_instance, serial_instance, num_episodes, learning_steps, display_count, directionModel)

    # results logging class
    results_instance = MKResults('./results/mk_data.csv')

    # initialize learning model and start training
    model_params = None # './params/mkEpisodes_480.params'
    mario_kart = MKNet(serial_instance, action_server_instance, env_instance, model_params, results_instance)
    mario_kart.play()

if __name__ == '__main__':
    main()