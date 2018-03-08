import keyboard
import time
                
def turn_left(event):
    print('left')
    pass

def turn_right(event):
    print('right')        
    pass

def accelerate(event):
    print('accelerate')        
    pass

def main():
    keyboard.hook_key('space', accelerate)
    keyboard.hook_key('left', turn_left)
    keyboard.hook_key('right', turn_right)

    while True:
        pass

if __name__ == '__main__':
    main()