# object_oriented_fer
Fer in a more object oriented way

# logger.py
Logging class allowing uniform output.
  Features:
    debug mode (optional)
    timestamp appended to message
    rediect standard error and standard out to files
  Note:
    mesages loggged with an E for error will only be printed out to the current standard out not the standard error

## example usage
  logger = logger.Logger(debug=1, log_file='out.txt', error_file='err.txt')
  
  logger.put_msg('D', 'Debug message', name=__file__, line=68)
  
  logger.put_msg('I', 'Information message', name=__file__, line=69)
  
  logger.put_msg('E', 'Error message', name=__file__, line=70)

# images.py
Image classes
  parent Class: Image
  children: CkImage, JaffeImage
    
 # data.py
  Loads the images into lists of Image objects
  
 # ck.py
  Driver for processing CK+ images
  
  
