import os
import sys



def generate_script(commands, matsize):

    fname = "cycle_" + str(matsize) + ".sh"
    writer = open(fname, 'w')

    for command in commands:
        writer.write(command[0] + ' > ' + 'CYCLE' +str(matsize)+ '_' +str(command[1])+'.txt' + '\n')
        writer.write('wait \n')
        writer.write('######################## \n')

    writer.close()
    print('Script generated.. ')


def generate_commands(low, high, command, matsize):

    sizes = []
    for i in range(low, high+1):
        sizes.append(i)

    commands = []

    for size in sizes:
        commands.append([command + ' -e' + str(size), size])

    generate_script(commands, matsize)
    




def main():

    low = 40
    high = 350
    matsize = 40

    command = "./gpu_perman -f erdos_real/" + str(matsize) + "_0.90_2.mtx -p35 -k4"

    generate_commands(low, high, command, matsize)




if __name__ == "__main__":

    main()
