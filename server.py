from openreward.environments import Server

from corebench import CoreBenchHard

if __name__ == "__main__":
    Server([CoreBenchHard]).run()
