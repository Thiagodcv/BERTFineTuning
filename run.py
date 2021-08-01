"""
The run module. 
"""
from preprocessing import preprocess
from experiment import cross_validation

def main():
    cross_validation.cross_validation(k=5, num_hyp=40, hyp_epochs=20, patience=10)

if __name__ == "__main__":
    main()


