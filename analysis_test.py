<<<<<<< HEAD
import analysis

data = analysis.load_csv('data.txt')
print(analysis.basic_stats(0, 'debug', 0))
print(analysis.basic_stats(data, "column", 0))
print(analysis.basic_stats(data, "row", 0))
print(analysis.z_score(10, analysis.basic_stats(data, "column", 0)[0],analysis.basic_stats(data, "column", 0)[3]))
print(analysis.histo_analysis(data[0]))
print(analysis.histo_analysis_2(data[0], 0.01, -1, 1))
print(analysis.stdev_z_split(3.3, 0.2, 0.1, -5, 5))

game_nc_entities = analysis.nc_entities(["cube", "cube", "ball"], [0, 1, 2], [[0, 0.5], [1, 1.5], [2, 2]], ["1;1;1;10', '2;1;1;20", "r=0.5, 5"], ["1", "1", "0"])
game_nc_entities.append("cone", 3, [1, -1], "property", "effect")
game_nc_entities.edit(2, "sphere", 10, [5, -5], "new prop", "new effect")
print(game_nc_entities.search(10))
print(game_nc_entities.debug())

game_obstacles = analysis.obstacles(["wall", "fortress", "castle"], [0, 1, 2],[[[10, 10], [10, 9], [9, 10], [9, 9]], [[-10, 9], [-10, -9], [-9, -10]], [[5, 0], [4, -1], [-4, -1]]] , [0, 0.01, 10])
game_obstacles.append("bastion", 3, [[50, 50], [49, 50], [50, 49], [49, 49]], 75)
game_obstacles.edit(0, "motte and bailey", "null", [[10, 10], [9, 10], [10, 9], [9, 9]], 0.01)
print(game_obstacles.search(0))
print(game_obstacles.debug())

game_objectives = analysis.objectives(["switch", "scale", "climb"], [0,1,2], [[0,0],[1,1],[2,0]], ["0,1", "1,1", "0,5"])
game_objectives.append("auto", 3, [0, 10], "1, 10")
game_objectives.edit(3, "null", 4, "null", "null")
print(game_objectives.search(4))
print(game_objectives.debug())
=======
import analysis

data = analysis.load_csv('data.txt')
print(analysis.basic_stats(0, 'debug', 0))
print(analysis.basic_stats(data, 1, 0))
print(analysis.basic_stats(data, 2, 0))
print(analysis.z_score(10, analysis.basic_stats(data, 1, 0)[0],analysis.basic_stats(data, 1, 0)[3]))
print(analysis.histo_analysis(data[0]))
print(analysis.histo_analysis_2(data[0], 0.01, -1, 1))
print(analysis.stdev_z_split(3.3, 0.2, 0.1, -5, 5))

x = analysis.objectives(["switch", "scale", "climb"], [0,1,2], [[0,0],[1,1],[2,0]], ["0,1", "1,1", "0,5"])
>>>>>>> 18189b45b17228228e192c270c12c3cc2cb7f8cf
