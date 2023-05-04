#Импортирование необходимых модулей из библиотеки NEAT
import os
import neat

#Определение входных и выходных данных XOR для обучения.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [0.0, 1.0, 1.0, 0.0]

# Приспособленность каждого генома в популяции
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Создание нейронной сети для каждого генома
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 4
        
        # Выполнить итерацию по парам ввода-вывода XOR
        for xi, xo in zip(xor_inputs, xor_outputs):
            # Активация входных значений
            output = net.activate(xi)
            
            # вычисление ошибки
            fitness -= (output[0] - xo) ** 2
        
        # На основе ошибки устанавливается приспособленность генома
        genome.fitness = fitness

# Запускает процесса нейроэволюции с использованием указанного файла конфигурации
def run_neuroevolution(config_file):
    # Загрузка конфигурации
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Создание популяции
    pop = neat.Population(config)

    # Отчёты для отображения прогресса и стат. информации
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Запуск процесса нейроэволюции на 300 поколений
    winner = pop.run(eval_genomes, 300)

    # Вывод лучшего генома
    print('\nBest genome:\n{!s}'.format(winner))

    # Показать выходные данные наиболее подходящего генома по сравнению с данными обучения.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

if __name__ == '__main__':
    # Файл с конфигурацией
    config_path = 'config-xor.txt'
    run_neuroevolution(config_path)
