import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """
    Runs a system command and ensures it succeeds.

    Args:
        command (list): The command to be executed as a list of strings.
        shell (bool, optional): Whether to run the command in the shell. Defaults to False.

    Raises:
        subprocess.CalledProcessError: If the command fails to execute successfully.
    """
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    """
    Runs the inference process using the specified model and arguments.

    This function determines the path to the main executable (llama-cli) based on the operating system,
    then constructs the command line arguments to be passed to the executable. It then calls the
    `run_command` function to execute the command and run the inference process.
    """
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        "-b", "1"
    ]
    run_command(command)

def signal_handler(sig, frame):
    """
    Handles the Ctrl+C signal (SIGINT) to gracefully exit the program.

    Args:
        sig (int): The signal number.
        frame (frame): The current stack frame.
    """
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict when generating text", required=False, default=128)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature, a hyperparameter that controls the randomness of the generated text", required=False, default=0.8)
    args = parser.parse_args()

    # Run the inference process
    run_inference()
