from argparse import ArgumentParser
import os
import json

def main(args):
    
    tokens = args.tokens.split(',')
    
    # initiate dict to store token counts results
    token_counts = {token : 0 for token in tokens}
    
    files = []
    for _,_,filenames in os.walk(args.input_dir):
        files.extend(filenames)
    
    for file in files:
        
        with open(file, 'r') as file:
            file_string = file.read().replace('\n','')
        
        for k, v in token_counts.items():
            token_counts[k] += file_string.count(k)
    
    output_file = 'token_counts.json'
    with open(os.path.join(args.output_dir, output_file), 'w') as file:
        json.dump(token_counts, file)
        
    print(f"Processing completed. Results are stored in {output_file}.")


if __name__ == "__main__":
    
    # Parse common arguments
    parser = ArgumentParser()
    parser.add_argument('--tokens', type=str, required=True, help="specify which tokens to count in input files")
    parser.add_argument('--input-dir', type=str, default="/opt/ml/processing/input_data", help="local dir with files for processing")
    parser.add_argument('--output-dir', type=str, default="/opt/ml/processing/processed_data", help="local dir with files for processing")
    args = parser.parse_args()
    
    main(args)