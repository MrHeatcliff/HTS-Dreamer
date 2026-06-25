import sys

from . import main_hts


def main(argv=None):
  argv = list(sys.argv[1:] if argv is None else argv)
  return main_hts.main(argv + ['--agent.hts.impl', 'hts1'])


if __name__ == '__main__':
  main()
