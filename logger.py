class Logger:
  VERBOSE = 0
  INFO = 1
  WARN = 2
  ERROR = 3
  FATAL = 4
  LAST = 5
  _log_level_dict = []
  _label = ["VERBOSE", "INFO", "WARN", "ERROR", "FATAL"]
  _overall_log_level = INFO
  @staticmethod
  def setLogLevel(logLevel, cate=None):
    if cate == None:
      Logger._overall_log_level = logLevel
    else:
      Logger._log_level_dict[cate] = logLevel

  @staticmethod
  def log(cate, level, message):
    if level >= Logger.LAST:
      return
    if cate in Logger._log_level_dict:
      if Logger._log_level_dict[cate] < level:
        return
    elif level < Logger._overall_log_level:
      return
    print "[%s][%s]: %s" % (cate, Logger._label[level], str(message))

  @staticmethod
  def verbose(cate, message):
    Logger.log(cate, Logger.VERBOSE, message)

  @staticmethod
  def info(cate, message):
    Logger.log(cate, Logger.INFO, message)

  @staticmethod
  def warn(cate, message):
    Logger.log(cate, Logger.WARN, message)

  @staticmethod
  def error(cate, message):
    Logger.log(cate, Logger.ERROR, message)

  @staticmethod
  def fatal(cate, message):
    Logger.log(cate, Logger.FATAL, message)