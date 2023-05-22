from options import Options
from solver import Solver

def main():

    args = Options().parse()
    solver = Solver(args)
    solver.run()

if __name__ == '__main__':
    main()
