'''
To-do list:
connect different chunks (if possible) before hand, then work
issue with district 1, figure out where it is going wrong
figure out how to read files (for centers)
figure out zoom meaning (bounding boxes)
make website
'''
import timeit
import urllib.request
import zipfile as zf
import geopandas
import pysal
import networkx
import shapely
from collections import Counter
import utm
import os
from past import autotranslate
import shapefile
from json import dumps, load
from github import Github, InputGitTreeElement
from pprint import pprint
autotranslate('geo_file_conv')
from geo_file_conv import shp_to_kml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

start = timeit.default_timer()
#graph stuff
def filtered(list, filter):
    new = []
    for i in list:
        if i in filter:
            new.append(i)
    return new

def subgraph(graph, vS):
    return {k: filtered(graph[k], vS) for k in vS}



#Next breadth first search step
def nextStep(graph, last):
    n = set()
    for i in last:
        for j in graph[i]:
            if j not in last:
                n.add(j)
    return n


'''
#Breadth first search
def bfs(graph,start):
  v = list(graph.keys())
  distances = [-1]*len(v)
  distances[v.index(start)]=0
  step = 1
  last = [start]
  while -1 in distances:
    last = nextStep(graph,last)
    for i in last:
      if distances[v.index(i)]==-1:
        distances[v.index(i)]=step
    step+=1
    if step>=len(v):
      break
  return dict(zip(v,distances))
'''


def bfs(graph, src):
    distance = {src: 0}
    visited = set()
    q = []
    q.append(src)
    visited.add(src)
    while len(q) != 0:
        current = q.pop(0)
        for dest in graph[current]:
            if dest not in visited:
                visited.add(dest)
                distance[dest] = distance[current] + 1
                q.append(dest)
    return distance


#Complete BFS
def matrixBFS(graph):
    m = []
    for i in list(graph.keys()):
        r = []
        b = bfs(graph, i)
        for j in list(graph.keys()):
            if j in b:
                r.append(b[j])
            else:
                r.append(-1)
        m.append(r)
    return m


def path(graph, start, end, length):
    current = [[start]]
    matG = matrixBFS(graph)
    vG = list(graph.keys())
    for i in range(length):
        nStep = []
        for j in current:
            n = nextStep(graph, [j[-1]])
            for k in n:
                nextOne = j + [k]
                last = nextOne[-1]
                if matG[vG.index(last)][vG.index(
                        end)] == length + 1 - len(nextOne):
                    nStep.append(nextOne)
        current = nStep
    good = []
    for k in current:
        if k[-1] == end:
            good.append(list(k))
    return good


#For determining subgraph



def boundary(graph, sub):
    vS = list(sub.keys())
    b = []
    for v in vS:
        connections = graph[v]
        isBoundary = False
        for j in connections:
            if j not in vS:
                isBoundary = True
                break
        if isBoundary:
            b.append(v)
    return b


#determines is graph is subgraph
def isSubgraph(graph, sub):
    #vertices are subset
    vG = list(graph.keys())
    vS = list(sub.keys())
    for v in vS:
        if v not in vG:
            return False
    filterG = {k: filtered(graph[k], vS) for k in vS}
    for v in vS:
        for i in sub[v]:
            if i not in filterG[v]:
                return False
        if len(filterG[v]) != len(sub[v]):
            return False
    return True


#determines if graph is convex subgraph
def isConvSub(graph, sub):
    if not isSubgraph(graph, sub):
        return False
    vG = list(graph.keys())
    vS = list(sub.keys())
    matG = matrixBFS(graph)
    matS = matrixBFS(sub)
    if matS == [] or matG == []:
        return False
    for i in vS:
        for j in vS:
            if matG[vG.index(i)][vG.index(j)] != matS[vS.index(i)][vS.index(
                    j)]:
                return False
    return True


#Doesn't take into consideration population of census blocks / VTDs
def convexnessSimple(graph, sub, matG=[], matS=[]):
    '''
  if not isSubgraph(graph,sub):
    return 0
  '''
    if len(list(sub.keys()))==1:
        return 1
    vG = list(graph.keys())
    vS = list(sub.keys())
    vertG = [vG.index(i) for i in vS]

    if matG == []:
        matG = matCT
    matS = matrixBFS(sub)
    sumS = sum([sum(matS[i]) for i in range(len(vS))])
    sumG = 0
    for u in vertG:
        for v in vertG:
            sumG += matG[u][v]
    if matG == []:
        return 0
    if matS == []:
        return 0
    '''for i in vS:
    for j in vS:
      sumG+=matG[vG.index(i)][vG.index(j)]
      sumS+=matS[vS.index(i)][vS.index(j)]
  '''
    if sumS == 0:
        print(sub)
    return sumG / sumS


def completeGraph(n):
    graph = {}
    for i in range(1, n + 1):
        key = i
        value = list(range(1, n + 1))
        value.remove(i)
        graph[key] = value
    return graph


def cycleGraph(n):
    graph = {}
    for i in range(1, n + 1):
        key = i
        value = [i - 1, i + 1]
        if i == 1:
            value = [n, 2]
        if i == n:
            value = [n - 1, 1]
        graph[key] = value
    return graph


def pathGraph(n):
    graph = {}
    for i in range(1, n + 1):
        key = i
        value = [i - 1, i + 1]
        if i == 1:
            value = [2]
        if i == n:
            value = [n - 1]
        graph[key] = value
    return graph




def all_edges(vertices):
    l = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if i != j:
                l.append([vertices[i], vertices[j]])
    return l


def random_graph(v, numE):
    allE = all_edges(v)
    edges = random.sample(allE, numE)
    graph = {}
    for i in v:
        graph[i] = []
    for e in edges:
        start = e[0]
        end = e[1]
        x = graph[start]
        x.append(end)
        graph[start] = x
        y = graph[end]
        y.append(start)
        graph[end] = y
    return graph


def addVertices(graph, sub, vs):
    subV = list(sub.keys())
    subV += vs
    return subgraph(graph, subV)


def removeVertices(graph, sub, vS):
    subV = list(sub.keys())
    for v in vS:
        subV.remove(v)
    return subgraph(graph, subV)


def avgDistance(pt, graph, sub, matG):
    vG = list(graph.keys())
    vS = list(sub.keys())
    ind = [vG.index(v) for v in vS]
    total = sum([matG[vG.index(pt)][i] for i in ind])
    return total / len(vS)


def avgDistLine(line, graph, sub, matG):
    vS = set(sub.keys())
    vG = set(graph.keys())

    newPts = []
    for v in line:
        if v not in vS:
          newPts.append(v)
    return sum([avgDistance(pt, graph, sub, matG)for pt in newPts]) / len(newPts)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def minConvex(graph, sub):
    notIncluded = []
    if not isConvSub(graph, graph):
        return {}
    if list(sub.values()) == [[]] * len(list(sub.values())):
        return {}
    for i in list(graph.keys()):
        if i not in list(sub.keys()):
            notIncluded.append(i)
    power = list(list(i) for i in powerset(notIncluded))
    for v in power:
        if isConvSub(graph, addVertices(graph, sub, v)):
            return addVertices(graph, sub, list(v))
    return {}


def random_subgraph(graph, numV):
    vS = random.sample(list(graph.keys()), numV)
    return subgraph(graph, vS)


def nextGreed(graph, sub):
    print('starting next step')
    maxiCon = 0
    bestAdd = []
    count = 1
    for i in nextStep(graph, list(sub.keys())):
        conI = convexnessSimple(graph, addVertices(graph, sub, [i]))
        if conI > maxiCon:
            maxiCon = conI
            bestAdd = [i]
        print(count)
        count += 1
    return [maxiCon, bestAdd]


def lastGreed(graph, sub):
    print('starting next step')
    maxiCon = 0
    bestAdd = []
    count = 1
    for i in boundary(graph, sub):
        r = removeVertices(graph, sub, [i])
        m = matrixBFS(r)
        if m == []:
            conI = 0
        else:
            conI = convexnessSimple(graph, removeVertices(graph, sub, [i]))
        if conI > maxiCon:
            maxiCon = conI
            bestAdd = [i]
        print(count)
        count += 1
    return [maxiCon, bestAdd]


def greedyMin(graph, sub):
    print('good')
    '''
  if not isConvSub(graph,graph):
    return {}
  '''
    current = sub
    print('good1')
    con = convexnessSimple(graph, sub)
    print('good2')
    count = len(list(current.keys()))
    if list(sub.values()) == [[]] * len(list(sub.values())):
        return {}
    while count <= len(list(graph.keys())):
        if con == 1:
            return current
        n = nextGreed(graph, current)
        if n[0] < con:
            return current
        current = addVertices(graph, current, n[1])
        con = n[0]
        count = len(list(current.keys()))
        print(n)
        f = open('steps.txt', 'a')
        f.write(str(n))
        f.write('\n')
        f.close()
    return graph


def greedyMax(graph, sub):
    print('good')
    '''
  if not isConvSub(graph,graph):
    return {}
  '''
    current = sub
    print('good1')
    con = convexnessSimple(graph, sub)
    print('good2')
    count = len(list(current.keys()))
    if list(sub.values()) == [[]] * len(list(sub.values())):
        return {}
    while count >= 0:
        if con == 1:
            return current
        n = lastGreed(graph, current)
        if n[0] < con:
            return current
        current = removeVertices(graph, current, n[1])
        con = n[0]
        count = len(list(current.keys()))
        print(n)
        f = open('steps.txt', 'a')
        f.write(str(n))
        f.write('\n')
        f.close()
    return graph


def biggestDiff(graph, sub, matG, matS):
    gDict = {}
    sDict = {}
    vG = list(graph.keys())
    vS = list(sub.keys())
    for v in range(len(vS)):
        sDict[vS[v]] = matS[v]
        gDict[vS[v]] = [matG[vG.index(vS[v])][vG.index(v2)] for v2 in vS]
    allDiff = []
    bigDiff = 0
    bigRatio = 0
    vals = []
    indices = []
    for v in vS:
        for v2 in vS:
            sVal = (sDict[v])[vS.index(v2)]
            gVal = (gDict[v])[vS.index(v2)]
            gap = abs(sVal - gVal)
            if gVal != 0:
                ratio = max(sVal / gVal, -1*gVal/sVal)
            else:
                ratio = 0
            if gap!=0:
              allDiff.append([sVal,gVal])
            if gap > bigDiff:
                bigDiff = gap
                bigRatio = ratio
                vals = [sVal, gVal]
                indices = [v, v2]

            if gap == bigDiff and ratio > bigRatio:
                bigDiff = gap
                bigRatio = ratio
                vals = [sVal, gVal]
                indices = [v, v2]
    #print(allDiff)
    return [indices, bigDiff, vals]


def bestLine(graph, sub, matG, matS):
  print('computing best line')
  bD = biggestDiff(graph, sub, matG, matS)
  lines = path(graph, bD[0][0], bD[0][1], bD[2][1])
  bestL = []
  minNew = float('inf')
  bestAvg = 0
  newStuff = []
  print('found all lines: '+str(len(lines)))
  for l in lines:
        stuff = []
        new = 0
        for i in l:
          if i not in set(sub.keys()):
            new += 1
            stuff.append(i)
        #a = avgDistLine(l, graph, sub, matG)
        a = convexnessSimple(graph,addVertices(graph,sub,l),matG,matrixBFS(addVertices(graph,sub,l)))
        if new < minNew:
          minNew = new
          bestL = l
          bestAvg = a
          newStuff = stuff
        if new == minNew and a > bestAvg:
          minNew = new
          bestL = l
          bestAvg = a
          newStuff = stuff
  print('found best line')
  #print(minNew)
  return [bestL, minNew, bD, newStuff]

def makeNonHole(graph,sub):
  p  = partitionsLeft(graph,sub)
  m = max([len(i) for i in p])
  good = []
  for i in p:
    if len(list(i.keys()))!=m:
      good+=i
  return addVertices(graph,sub,good)


def bestLineHole(graph, sub, matG, matS):
  print('computing best line')
  bD = biggestDiff(graph, sub, matG, matS)
  lines = path(graph, bD[0][0], bD[0][1], bD[2][1])
  bestFill = []
  minNew = float('inf')
  bestAvg = 0
  newStuff = []
  print('found all lines')
  length = len(lines)
  count = 1
  print(length)

  for l in lines:
      nonHole = makeNonHole(graph,addVertices(graph,sub,l))
      #print(nonHole)
      stuff = []
      new = 0
      for i in list(nonHole.keys()):
        if i not in set(sub.keys()):
          new += 1
          stuff.append(i)
#      print(new)
        #a = avgDistLine(l, graph, sub, matG)
      a = convexnessSimple(graph,nonHole,matG,matrixBFS(nonHole))
      if new < minNew:
        minNew = new
        bestFill = nonHole
        bestAvg = a
        newStuff = stuff
      if new == minNew and a > bestAvg:
        minNew = new
        bestFill = nonHole
        bestAvg = a
        newStuff = stuff
     # print(count)
      count+=1
  print('found best line')
#  print(minNew)
  return [bestFill, minNew, bD, newStuff]

def lineMin(graph, sub, chunkNum=0,cap=1):
    print('starting lineMin')
    ys = []
    current = sub
    matG = matrixBFS(graph)
    matS = matrixBFS(sub)
    conv = convexnessSimple(graph, current, matG, matrixBFS(current))
    ys.append(conv)
    count = 1
    while conv < cap:
        line = bestLine(graph, current, matG, matrixBFS(current))
        #print(line)

        current = addVertices(graph,current,line[0])
        '''if stepByStep!=False:
            writeShapeFile(current,stepByStep,stepByStep[:-4]+'Chunk'+str(chunkNum)+'Step'+str(count)+'.shp')
            ShpToJSON(stepByStep[:-4]+'Chunk'+str(chunkNum)+'Step'+str(count)+'.shp',stepByStep[:-4]+'Chunk'+str(chunkNum)+'Step'+str(count)+'.json')
            postGithub(stepByStep[:-4]+'Chunk'+str(chunkNum)+'Step'+str(count)+'.json')'''
        count+=1
        #print(current)
        conv = convexnessSimple(graph, current, matG, matrixBFS(current))
        print(conv)
        #print('holes:')
        #print([len(list(i.keys())) for i in partitionsLeft(graph,current)])
        ys.append(conv)
    xs = list(range(len(ys)))
    #plt.plot(xs, ys, 'r')
    #plt.savefig('graph.png')
    print([len(list(i.keys())) for i in partitionsLeft(graph,current)])
    return current

def lineMinHole(graph, sub, cap=1):
    print('starting lineMinHole')
    ys = []
    current = sub
    matG = matrixBFS(graph)
    matS = matrixBFS(sub)
    conv = convexnessSimple(graph, current, matG, matrixBFS(current))
    ys.append(conv)
    while conv < cap:
        line = bestLineHole(graph, current, matG, matrixBFS(current))
        #print(line)
        current = line[0]
        conv = convexnessSimple(graph, current, matG, matrixBFS(current))
        print(conv)
        #print('holes:')
        #print([len(list(i.keys())) for i in partitionsLeft(graph,current)])
        ys.append(conv)
    xs = list(range(len(ys)))
    #plt.plot(xs, ys, 'r')
    #plt.savefig('graph.png')
    return makeNonHole(graph,current)


testG = {
    1: [2, 5],
    2: [1, 3, 6],
    3: [2, 4],
    4: [3, 5, 6],
    5: [1, 4],
    6: [2, 4]
}
testS = {1: [2], 2: [3, 1], 3: [4, 2], 4: [3]}
graph = {
    "a": ["c"],
    "b": ["c", "e"],
    "c": ["a", "b", "d", "e"],
    "d": ["c"],
    "e": ["c", "b"],
}

disSize = []
time1 = []
time2 = []
len1 = []
len2 = []

g = {'a': ['b'], 'b': ['a'], 'c': []}

def exclusiveNextStep(graph, last,bad):
    n = set()
    for i in last:
        for j in graph[i]:
            if j not in last and j not in bad:
                n.add(j)
    return n
def createSubgraph(graph,sub,start):
  verts = [start]
  while len(list(exclusiveNextStep(graph,verts,list(sub.keys()))))!=0:
    verts+=list(exclusiveNextStep(graph,verts,list(sub.keys())))
  return subgraph(graph,verts)

def hasNoHoles(graph, sub):
    vG = list(graph.keys())
    vS = list(sub.keys())
    notIn = []
    for v in vG:
        if v not in vS:
            notIn.append(v)
    g = createSubgraph(graph,sub,notIn[0])
    for i in g.keys():
      notIn.remove(i)
    return notIn==[]

def partitionsLeft(graph,sub):
  vG = list(graph.keys())
  vS = list(sub.keys())
  notIn = []
  for v in vG:
    if v not in vS:
      notIn.append(v)
  p = []
  while len(notIn)!=0:
    g = createSubgraph(graph,sub,notIn[0])
    p.append(g)
    for i in g.keys():
      notIn.remove(i)
  return p

'''  vG = list(graph.keys())
  vS = list(sub.keys())
  notIn = []
  for v in vG:
    if v not in vS:
      notIn.append(v)
  p = []
  while len(sub)!=0:
    g = createSubgraph(graph,subgraph(graph,notIn),v[0])
    p.append(g)
    for i in g.keys():
      vS.remove(i)
  return p
'''
def partitionsOf(graph,sub):

    vG = list(graph.keys())
    vS = list(sub.keys())
    notIn = []
    for v in vG:
        if v not in vS:
            notIn.append(v)
    return partitionsLeft(graph,subgraph(graph,notIn))


def numBadVerts(graph,sub):
  parts = partitionsLeft(graph,sub)
  sizes = [len(list(p.keys())) for p in parts]
  return sizes


def intBoundary(graph,subgraph):
  boundVerts = []
  for i in list(subgraph.keys()):
    for j in graph[i]:
      if j not in list(subgraph.keys()):
        boundVerts.append(i)
        break
  return boundVerts

def sharedBoundary(sub1,sub2):
  shared = []
  for i in list(sub1.keys()):
    if i in list(sub2.keys()):
      shared.append(i)
  return shared

def holeness(graph,sub):
  holes = []
  for i in partitionsLeft(graph,sub):
    holes.append(len(i.keys()))
  return (sum(holes)-max(holes))/len(sub.keys())



start = timeit.default_timer()


#Step 1: make graph from shp
#shp to graph
def get_list_of_data(filepath, col_name, geoid=None):
    """Pull a column data from a shape or CSV file.
    :filepath: The path to where your data is located.
    :col_name: A list of the columns of data you want to grab.
    :returns: A list of the data you have specified.
    """
    # Checks if you have inputed a csv or shp file then captures the data
    if filepath.split('.')[-1] == 'csv':
        df = pandas.read_csv(filepath)
    elif filepath.split('.')[-1] == 'shp':
        df = geopandas.read_file(filepath)
    elif filepath.split('.')[-1] == 'geojson':
        df = geopandas.read_file(filepath)

    if geoid is None:
        geoid = "sampleIndex"
        df[geoid] = range(len(df))

    data = pandas.DataFrame({geoid: df[geoid]})
    for i in col_name:
        data[i] = df[i]
    return data


def add_data_to_graph(df, graph, col_names, id_col=None):
    """Add columns of a dataframe to a graph based on ids.
    :df: Dataframe containing given column.
    :graph: NetworkX object containing appropriately labeled nodes.
    :col_names: List of dataframe column names to add.
    :id_col: The column name to pull graph ids from. The row from this id will
             be assigned to the corresponding node in the graph. If `None`,
             then the data is assigned to consecutive integer labels 0, 1, ...,
             len(graph) - 1.
    """
    if id_col:
        for row in df.itertuples():
            node = getattr(row, id_col)
            for name in col_names:
                data = getattr(row, name)
                graph.nodes[node][name] = data
    else:
        for i, row in enumerate(df.itertuples()):
            for name in col_names:
                data = getattr(row, name)
                graph.nodes[i][name] = data


def neighbors_with_shared_perimeters(neighbors, df):
    vtds = {}

    for shape in neighbors:
        vtds[shape] = {}

        for neighbor in neighbors[shape]:
            shared_perim = df.loc[shape, "geometry"].intersection(
                df.loc[neighbor, "geometry"]).length
            vtds[shape][neighbor] = {'shared_perim': shared_perim}

    return vtds


def add_boundary_perimeters(graph, neighbors, df):
    all_units = df['geometry']
    # creates one shape of the entire state to compare outer boundaries against
    inter = geopandas.GeoSeries(shapely.ops.cascaded_union(all_units).boundary)

    # finds if it intersects on outside and sets
    # a 'boundary_node' attribute to true if it does
    # if it is set to true, it also adds how much shared
    # perimiter they have to a 'boundary_perim' attribute
    for node in neighbors:
        graph.node[node]['boundary_node'] = inter.intersects(
            df.loc[node, "geometry"]).bool()
        if inter.intersects(df.loc[node, "geometry"]).bool():
            graph.node[node]['boundary_perim'] = float(
                inter.intersection(df.loc[node, "geometry"]).length)
    return graph


def add_areas(graph, df):
    df = reprojected(df)
    for node in graph.nodes:
        graph.nodes[node]['area'] = df.loc[node, "geometry"].area
    return graph


def add_centroids(graph, df):
    for node in graph.nodes:
        graph.nodes[node]['centroid'] = df.loc[node, "geometry"].centroid


def get_neighbors(df, adjacency_type):
    print('getting there')
    if adjacency_type == 'queen':
        return pysal.weights.Queen.from_dataframe(
            df, geom_col="geometry").neighbors
    elif adjacency_type == 'rook':
        return 'uh oh'
        print('closer')
        return pysal.weights.Rook.from_dataframe(
            df, geom_col="geometry").neighbors
    else:
        raise ValueError('adjacency_type must be rook or queen.')


def add_columns(graph, cols_to_add, df, geoid_col):
    if cols_to_add is not None:
        data = pandas.DataFrame({x: df[x] for x in cols_to_add})
        if geoid_col is not None:
            data[geoid_col] = df.index
        add_data_to_graph(data, graph, cols_to_add, id_col=geoid_col)


def construct_graph_from_df(df,  adjacency_type, geoid_col=None, cols_to_add=None):
    """Construct initial graph from information about neighboring VTDs.
    :df: Geopandas dataframe.
    :returns: NetworkX Graph.
    """
    # reproject to a UTM projection for accurate areas and perimeters in meters
    #df = reprojected(df)
    if 'GEOID' in df.columns:
        geoid_col = 'GEOID'
    elif 'GEOID10' in df.columns:
        geoid_col = 'GEOID10'
    print('1')
    if geoid_col is not None:
        df = df.set_index(geoid_col)
    print('2')
    # Generate rook or queen neighbor lists from dataframe.
    neighbors = get_neighbors(df, adjacency_type)
    print('3')
    vtds = neighbors_with_shared_perimeters(neighbors, df)
    print('4')
    graph = networkx.from_dict_of_dicts(vtds)
    print('5')
    add_boundary_perimeters(graph, neighbors, df)
    print('6')
    add_areas(graph, df)
    print('7')
    add_columns(graph, cols_to_add, df, geoid_col)
    print('8')
    return graph


def construct_graph_from_json(jsonData):
    """Construct initial graph from networkx.json_graph adjacency json format
    :jsonData: adjacency_graph data in json format
    :returns: networkx graph
    """
    return networkx.json_graph.adjacency_graph(jsonData)


def construct_graph_from_file(filename, adjacency_type=None, geoid_col=None, cols_to_add=None):
    """Constucts initial graph from either json(networkx adjacency_graph format) file
    or from a shapefile
    NOTE: at this point only supports the following 2 file formats:
    - ESRI shapefile
    - networkx.readwrite.json_data serialized json
    :filename: name of file to read
    :geoid_col: unique identifier for the data units to be used as nodes in the graph
    :cols_to_add: list of column names from file of data to be added to each node
    :returns: networkx graph
    """
    print('nice')
    if filename.split('.')[-1] == "json":
        mydata = json.loads(open(filename).read())
        graph = construct_graph_from_json(mydata)
        return graph
    elif filename.split('.')[-1] == "shp":
        print('nice2')
        print(filename)
        df = geopandas.read_file(filename)
        print('read')
        graph = construct_graph_from_df(
            df, adjacency_type, geoid_col, cols_to_add)
        return graph


def construct_graph(data_source, adjacency_type=None, geoid_col=None,
                    data_cols=None, data_source_type="filename"):
    """Constructs initial graph using the graph constructor that best
    matches the data_source and dataType formats
    :data_source: data from which to create graph (file name, graph object, json, etc)
    :geoid_col: name of unique identifier for basic data units
    :data_cols: any extra data contained in data_source to be added to nodes of graph
    :dataType: string specifying the type of data_source
    :returns: netwrokx graph
    """
    print('cool')
    if data_source_type == "filename":
        print('cool2')
        return construct_graph_from_file(data_source, adjacency_type, geoid_col, data_cols)

    elif data_source_type == "json":
        return construct_graph_from_json(data_source)

    elif data_source_type == "geo_data_frame":
        return construct_graph_from_df(data_source, adjacency_type, geoid_col, data_cols)


def get_assignment_dict(df, key_col, val_col):
    """Compute a dictionary from the given columns of the dataframe.
    :df: Dataframe.
    :key_col: Column name to be used for keys.
    :val_col: Column name to be used for values.
    :returns: Dictionary of {key: val} pairs from the given columns.
    """
    dict_df = df.set_index(key_col)
    return dict_df[val_col].to_dict()

def utm_of_point(point):
    return utm.from_latlon(point.y, point.x)[2]


def identify_utm_zone(df):
    df = df.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    utms = map(utm_of_point, df['geometry'].centroid)
    utm_counts = Counter(utms)
    # most_common returns a list of tuples, and we want the 0,0th entry
    most_common = utm_counts.most_common(1)[0][0]
    return most_common


def reprojected(df):
    utm = identify_utm_zone(df)
    return df
    #return df.to_crs(f"+proj=utm +zone={utm} +ellps=WGS84 +datum=WGS84 +units=m +no_defs")


def makeNiceGraph(f):
  graph = construct_graph(f,'queen','GEOID')
  graph = networkx.to_dict_of_lists(graph)
  return graph

#Step 2: Assign blocks
#block assign
def isNum(n):
  nums = '1234567890'
  if n=='':
      return False
  for i in n:
    if i not in nums:
      return False
  return True
def districtOne(blockGroup,f,length):
  districting = open(f,'r')
  d = {0:0}
  for line in districting:
    if line[:length]==blockGroup:
      district = line[-3:-1]
      value = 0
      if not isNum(district):
        value = 0
      else:
        value = int(district)
      if value in d.keys():
        k = d[value]
        k+=1
        d[value]=k
      else:
        d[value]=1
  return d
def allDis(ds,f,length):
  print('allDis')
  districting = open(f,'r')
  d = {}
  for dis in ds:
    d[dis]=districtOne(dis,f,length)
  return d
def maximize(allD):
  print('maximize')
  maxD = {}
  for i in allD.keys():
    d = allD[i]
    mD = ''
    mV = 0
    for j in d.keys():
      if d[j]>mV:
        mV = d[j]
        mD = j
    if mD!='':
        maxD[i]=mD
    else:
        maxD[i]=0
  return maxD

def allDistricting(maxi):
  dis = []
  maxiList = list(maxi.values())
  for i in range(max(maxiList)+1):
    dis.append([])
  for j in list(maxi.keys()):
    if maxi[j] in range(len(dis)):
      d=dis[maxi[j]]
      d.append(j)
      dis[maxi[j]]=d
  return dis
def doDistricting(ds,f,length):
  aD = allDis(ds,f,length)
  print('allDis')
  print(aD)
  maxi = maximize(aD)
  print('maximize')
  print(maxi)
  allD = allDistricting(maxi)
  return allD
def importantDis(allDis):
  important = []
  for i in range(1,len(allDis)):
    important+=allDis[i]
  return important

#Step 3: Do stuff with graph

#Step 4: Write stuff to shapefile
def writeShapeFile(dis,source,dest):
  df = geopandas.read_file(source)
  new = geopandas.GeoDataFrame()
  new = df[df.GEOID.isin(dis)]
  new.to_file(dest)

#Step 5: Turn shapefile into JSON
def ShpToJSON(shp,json):
   # read the shapefile
   reader = shapefile.Reader(shp)
   fields = reader.fields[1:]
   field_names = [field[0] for field in fields]
   buffer = []
   for sr in reader.shapeRecords():
       atr = dict(zip(field_names, sr.record))
       geom = sr.shape.__geo_interface__
       buffer.append(dict(type="Feature", \
        geometry=geom, properties=atr))

   # write the GeoJSON file
   geojson = open(json, "w")
   geojson.write(dumps({"type": "FeatureCollection",\
    "features": buffer}, indent=2) + "\n")
   geojson.close()

#Step 6: Turn shapefile into KML

#CSV to json
def parse(dis,name,basis):
  with open(basis, newline = '\n') as f:
    reader = csv.DictReader(f)
    fieldnames = []
    for row in reader:
      fieldnames = list(row.keys())
      break
  with open(basis,newline='\n') as f:
    reader = csv.DictReader(f)
    with open(str(name)+'.csv', mode='w') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()
      for row in reader:
        print(row['GEOID'])
        if row['GEOID'] in dis:
          writer.writerow(row)

def turnToJson(csvFile,name):
  features = []
  with open(csvFile, newline='\n') as csvfile:
      reader = csv.DictReader(csvfile, delimiter=',')
      for row in reader:
          latitude, longitude = map(float, (float(row['latitude']), float(row['longitude'])))
          features.append(
              Feature(
                  geometry = geojson.Point((longitude, latitude)),
                  properties = {
                    'GEOID': row['GEOID']
                }
            )
        )

  collection = geojson.FeatureCollection(features)
  with open(str(name)+'.json', "w") as f:
      f.write('%s' % collection)
#parse(['090010426001','090010426002'],'new','dis2BGhole.csv')
#turnToJson('new.csv','new')

#Turn back to shapefile
def turnToSHP(basis,dest,dis):
  df = geopandas.read_file(basis)
  new = geopandas.GeoDataFrame()
  new = df[df.GEOID.isin(dis)]
  new.to_file(dest)

def zipStuff(shpname,dest):
  print('zipping ',shpname, 'to ',dest)
  if os.path.exists(dest):
      os.remove(dest)
  with zf.ZipFile(dest, 'w') as myzip:
    myzip.write(shpname+'.shp')
    myzip.write(shpname+'.shx')
    myzip.write(shpname+'.dbf')
    myzip.write(shpname+'.cpg')
    myzip.write(shpname+'.prj')
  myzip.close()

def downloadShpFromUrl(url,file_name):
  print(url)
  print(file_name)
  exists = os.path.isfile(file_name)
  print(file_name,exists)
  if exists:
      return None
  f = urllib.request.urlopen(url)
  data = f.read()
  with open(file_name, "wb") as code:
    code.write(data)
    code.close()

def downloadShp(state,t,dest):
  stateToFip = {}
  abbToFip = {}

  fips = open('/Users/samflorin/Documents/Gerrymandering/stateFIPS.txt','r')
  state = state.lower()
  for line in fips:
    line = line.replace('\n','')
    line= line.split('	')
    stateToFip[line[0].lower()] = line[1]
    abbToFip[line[2].lower()] = line[1]
  extra = ''
  if t=='tabblock':
      extra = '10'
  if state in list(stateToFip.keys()):
    fip = stateToFip[state]
  elif state in list(abbToFip.keys()):
    fip = abbToFip[state]
  else:
    return 'Error'
  url = 'https://www2.census.gov/geo/tiger/TIGER2018/%s/tl_2018_%s_%s%s.zip' % (t.upper(),str(fip),t.lower(),extra)
  exists = os.path.isfile(dest)
  print(dest,exists)
  if exists:
      return [fip,url]
  downloadShpFromUrl(url,dest)
  return [fip,url]
def getGraphFromInfo(state,layer,dest):
  gFile = '/Users/samflorin/Documents/Gerrymandering/%s%sGraph.txt'%(state,layer)
  dShp = downloadShp(state,layer,dest)
  fip = dShp[0]
  slashes = dShp[1].split('/')
  shpStart = '/Users/samflorin/Documents/Gerrymandering/'+slashes[-1][:-4]
  exists = os.path.isfile(gFile)
  print(gFile,exists)
  if exists:
      with open(gFile) as f:
          g = load(f)
      return [g,shpStart+'.shp']
  zippy = zf.ZipFile(dest,'r')
  zippy.extractall('/Users/samflorin/Documents/Gerrymandering/')
  g = makeNiceGraph(shpStart+'.shp')
  with open(gFile,'w') as f:
      f.write(dumps(g))
  return [g,shpStart+'.shp']
'''url = 'https://www2.census.gov/geo/tiger/TIGER2018/BG/tl_2018_09_bg.zip'
req = urllib.request.Request(url)
try:
     urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
  print(e.code)
  print(e.read())

graphTract,shpName = getGraphFromInfo('ct','bg','CTtracts.zip')
print('found graph')
tractDistricts = doDistricting(list(graphTract.keys()),'BlockAssign_ST09_CT_CD.txt',len(list(graphTract.keys())[0]))
dictionary = {}
for i in range(1,6):
  for j in tractDistricts[i]:
    dictionary[j]=i

print('districts')

filled = lineMinHole(graphTract,subgraph(graphTract,tractDistricts[1]))
print(filled)
writeShapeFile(list(filled.keys()),shpName,'BGdis1filled.shp')
zipStuff('BGdis1filled','BGdis1filled.zip')
print('\n')
print('\n')
'''
def chunkDist(chunk1,chunk2,graph,matG):
    vG = list(graph.keys())
    v1 = list(chunk1.keys())
    v2 = list(chunk2.keys())

    minDist = 10**7
    pairs = []
    for i in v1:
        for j in v2:
            d = matG[vG.index(i)][vG.index(j)]

            if d<minDist and d!=-1:
                minDist=d
                pairs = [[i,j]]
            if d==minDist and d!=-1:
                pairs.append([i,j])
    return [minDist,pairs]
def bestConnector(pairs,minDist,graph,dis,matG):
    bestConn = []
    bestConv = 0
    for i in pairs:
        for j in path(graph,i[0],i[1],minDist):
            if convexnessSimple(graph,addVertices(graph,dis,j),matG)>bestConv:
                bestConn = addVertices(graph,dis,j)
    return bestConn
def connectChunks(chunks,graph,dis,matG):
    newState = dis
    mat = []
    matPairs = []
    for i in chunks:
        row = []
        rowPairs = []
        for j in chunks:
            cD = chunkDist(i,j,graph,matG)
            row.append(cD[0])
            rowPairs.append(cD[1])
        mat.append(row)
        matPairs.append(rowPairs)
    print(mat)
    spanTree = minimum_spanning_tree(csr_matrix(mat))
    print(spanTree)
    spanTree = spanTree.toarray().astype(int)
    print(spanTree)
    connGraph = {}
    for i in range(len(spanTree)):
        connGraph[i]=[]
    for i in range(len(spanTree)):
        for j in range(len(spanTree[i])):
            if spanTree[i][j]!=0 and spanTree[i][j]!=10**7:
                connGraph[i].append(j)
                connGraph[j].append(i)
    chunkChunks = partitionsOf(connGraph,connGraph)
    if len(chunkChunks)==1:
        newState = dis
        for i in range(len(spanTree)):
            for j in range(len(spanTree[i])):

                if spanTree[i][j]!=0:
                    newState = bestConnector(matPairs[i][j],mat[i][j],graph,newState,matG)
        return [newState]
    else:
        newState = []
        for i in chunkChunks:
            newChunks = [chunks[j] for j in list(i.keys())]
            nextVs=[]
            for i in newChunks:
                nextVs+=list(i.keys())
            newState.append(connectChunks(newChunks,graph,subgraph(graph,nextVs),matG)[0])
        return newState


def chunkConnectDis(state,district,dataType):
    zipFilename = '/Users/samflorin/Documents/Gerrymandering/'+state+dataType.capitalize()+'.zip'
    districtsFile = downloadBlockAssign(state)
    #print(districtsFile)
    numOfDistricts = getNumOfDistricts(state)
    #print(numOfDistricts)
    graph,shpName = getGraphFromInfo(state,dataType,zipFilename)
    waterDis = waterDistricts(shpName)
    pops = getPopOfBlock(state)
    graph = subgraph(graph,list(Counter(list(graph.keys()))-Counter(waterDis)))
    graphFile = '/Users/samflorin/Documents/Gerrymandering/%s%sGraph.txt'%(state,dataType)
    districts = doDistricting(list(graph.keys()),districtsFile,len(list(graph.keys())[0]))
    #print(len(districts))
    #print(districts[0])
    for i in districts[0]:
        disNei = []
        neighbors = graph[i]
        #print('neighbors:',neighbors)
        for j in neighbors:
            val = getDistrict(j,districts)
            #print('val:',val)
            if val!=0:
                disNei.append(val)
        if len(disNei)!=0:
            actual = round(sum(disNei)/len(disNei))
        else:
            actual = 0
        #print(actual)
        districts[0].remove(i)
        districts[actual].append(i)
    districts = ['blank for indexing']+[list(districts[i]) for i in range(1,len(districts))]
    for i in range(len(districts)):
        print(len(districts[i]))
        print(type(districts[i]))
    print('districts:',len(districts))
    print(numOfDistricts)
    goodDis = subgraph(graph,districts[district])
    parts = partitionsOf(graph,subgraph(graph,districts[district]))
    newDis = connectChunks(parts,graph,goodDis,matrixBFS(graph))
    #print(newDis)
    saveDis = '/Users/samflorin/Documents/Gerrymandering/%s%s%sChunkied.shp'%(state,district,dataType)
    writeShapeFile(newDis,shpName,saveDis)
    zipStuff(saveDis[:-4],saveDis[:-4]+'.zip')



'''get min distance between any two chunks
basically becomes TSP (c)'''

def downloadBlockAssign(state,name=''):

  stateToFip = {}
  abbToFip = {}

  fips = open('/Users/samflorin/Documents/Gerrymandering/stateFIPS.txt','r')
  state = state.lower()
  for line in fips:
    line = line.replace('\n','')
    line= line.split('	')
    stateToFip[line[0].lower()] = line[1]
    abbToFip[line[2].lower()] = line[1]

  if state in list(stateToFip.keys()):
    fip = stateToFip[state]
  elif state in list(abbToFip.keys()):
    fip = abbToFip[state]
  abbs = list(abbToFip.keys())
  allFips = list(abbToFip.values())
  abb = abbs[allFips.index(fip)].upper()
  if name =='':
    name = '/Users/samflorin/Documents/Gerrymandering/BlockAssign_ST%s_%s.zip'%(fip,abb)
  url = 'https://www2.census.gov/geo/docs/maps-data/data/baf/BlockAssign_ST%s_%s.zip'%(fip,abb)
  downloadShpFromUrl(url,name)
  zippy = zf.ZipFile(name,'r')
  zippy.extractall('/Users/samflorin/Documents/Gerrymandering/')

  return '/Users/samflorin/Documents/Gerrymandering/BlockAssign_ST%s_%s_CD.txt'%(fip,abb)

def getPopOfBlock(state,name=''):
  stateToFip = {}
  abbToFip = {}
  fips = open('/Users/samflorin/Documents/Gerrymandering/stateFIPS.txt','r')
  state = state.lower()
  for line in fips:
    line = line.replace('\n','')
    line= line.split('	')
    stateToFip[line[0].lower()] = line[1]
    abbToFip[line[2].lower()] = line[1]
  if state in list(stateToFip.keys()):
    fip = stateToFip[state]
  elif state in list(abbToFip.keys()):
    fip = abbToFip[state]
  url = 'https://www2.census.gov/geo/tiger/TIGER2010BLKPOPHU/tabblock2010_%s_pophu.zip'%(fip)
  if name =='':
    name = '/Users/samflorin/Documents/Gerrymandering/'+'tabblock2010_%s_pophu.zip'%(fip)
  downloadShpFromUrl(url,name)
  zippy = zf.ZipFile(name,'r')
  zippy.extractall('/Users/samflorin/Documents/Gerrymandering/')

  source = name[:-4]+'.shp'
  print(source)
  df = geopandas.read_file(source)
  print(df.columns)
  return dict(zip(list(df['BLOCKID10'].to_dict().values()),list(df['POP10'].to_dict().values())))

def union(a,b):
    u = b
    for i in a:
        if i not in u:
            u.append(a)
    return u



def centerAndBound(file):
    co1 = []
    co2 = []
    with open(file) as f:
        data = load(f)
    for i in range(len(data['features'])):
        if data['features'][i]['geometry']['type']=='Polygon':
            poly = data['features'][i]['geometry']['coordinates'][0]
            for j in range(len(poly)):
                co1.append(poly[j][0])
                co2.append(poly[j][1])
        elif data['features'][i]['geometry']['type']=='MultiPolygon':
            multi = data['features'][i]['geometry']['coordinates'][0]
            for j in range(len(multi)):
                poly = multi[j]
                for j in range(len(poly)):
                    co1.append(poly[j][0])
                    co2.append(poly[j][1])
    try:
        return [[sum(co1)/len(co1),sum(co2)/len(co2)],[max(co1)-min(co1),max(co2),min(co2)]]
    except:
        return 'ruh roh'


def getNumOfDistricts(state):
  stateToFip = {}
  abbToFip = {}
  fips = open('/Users/samflorin/Documents/Gerrymandering/stateFIPS.txt','r')
  state = state.lower()
  for line in fips:
    line = line.replace('\n','')
    line= line.split('	')
    stateToFip[line[0].lower()] = line[1]
    abbToFip[line[2].lower()] = line[1]

  if state in list(stateToFip.keys()):
    fip = stateToFip[state]
  elif state in list(abbToFip.keys()):
    fip = abbToFip[state]
  abbs = list(abbToFip.keys())
  allFips = list(abbToFip.values())
  abb = abbs[allFips.index(fip)]
  stateToDis = {}
  dis = open('/Users/samflorin/Documents/Gerrymandering/stateDistricts.txt','r')
  for line in dis:
    line = line.replace('\n','')
    line = line.split('	')
    stateAbb = line[1].lower()
    stateDis = int(line[-1])
    stateToDis[stateAbb] = stateDis
  return stateToDis[abb]

def waterDistricts(shpFile):
  print(shpFile)
  df = geopandas.read_file(shpFile)
  print(df.columns)
  if 'tabblock' in shpFile:
      return df.loc[df.ALAND10<=df.AWATER10,'GEOID10'].values[:]
  else:
      return df.loc[df.ALAND<=df.AWATER,'GEOID'].values[:]
  print(df.columns)

  #print(df.loc[df.ALAND==0,'GEOID'].values[:])



def getShape(state,datatype,district,typeOfConvexness,saveDistrict=False,zipFilename='',outputFile=''):
  if zipFilename=='':
    zipFilename = '/Users/samflorin/Documents/Gerrymandering/'+state+datatype.capitalize()+'.zip'
  if outputFile=='':
    outputFile = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+datatype.capitalize()+typeOfConvexness.capitalize()+'.shp'
  districtsFile = downloadBlockAssign(state)
  print(districtsFile)
  numOfDistricts = getNumOfDistricts(state)
  print(numOfDistricts)
  graph,shpName = getGraphFromInfo(state,datatype,zipFilename)
  waterDis = waterDistricts(shpName)
  graph = subgraph(graph,list(Counter(list(graph.keys()))-Counter(waterDis)))
  districts = doDistricting(list(graph.keys()),districtsFile,len(list(graph.keys())[0]))
  pops = getPopOfBlock(state)

  #print(districts)
  districts = ['blank for indexing']+[list(districts[i]) for i in range(1,len(districts))]
  for i in range(len(districts)):
      print(len(districts[i]))
      print(type(districts[i]))

  dictionary = {}
  for i in range(1,numOfDistricts+1):
    for j in districts[i]:
      dictionary[j]=i
  parts = partitionsOf(graph,subgraph(graph,districts[district]))
  lens = [len(i) for i in parts]
  if len(lens)!=1:
      return lens


  if typeOfConvexness.lower()=='hole':
    filled = lineMinHole(graph,subgraph(graph,districts[district]))
  elif typeOfConvexness.lower()=='line':
    filled = lineMin(graph,subgraph(graph,districts[district]))
  else:
    return 'invalid convexness type'
  if saveDistrict!=False:
    if saveDistrict == True:
      saveDistrict = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+datatype.capitalize()+'.shp'
    writeShapeFile(list(subgraph(graph,districts[district]).keys()),shpName,saveDistrict)
    zipStuff(saveDistrict[:-4],saveDistrict[:-4]+'.zip')
  writeShapeFile(list(filled.keys()),shpName,outputFile)
  zipStuff(outputFile[:-4],outputFile[:-4]+'.zip')
  newPops = {}
  l = len(list(filled.keys())[0])
  for i in list(pops.keys()):
    j = i[:l]
    if j not in list(newPops.keys()):
        newPops[j]=pops[i]
    else:
        k=newPops[j]
        k+=pops[i]
        newPops[j]=k
  ShpToJSON(outputFile,outputFile[:-3]+'json')
  postGithub(outputFile[:-3]+'json')
  s1 = 0
  for b in list(filled.keys()):
      if b in list(newPops.keys()):
          s1+=newPops[b]
  #sum([newPops[b] for b in list(filled.keys())])
  s2 = 0
  #sum([newPops[b] for b in districts[district]])
  for b in districts[district]:
      if b in list(newPops.keys()):
          s2+=newPops[b]
  return [s1/s2, outputFile]

def getDistrict(district,districting):
    for i in range(len(districting)):
        if district in districting[i]:
            return i
def singlify(list):
    new = []
    for i in list:
        if i not in new:
            new.append(i)
    return new

def completeState(state,dataType,typeOfConvexness):
    zipFilename = '/Users/samflorin/Documents/Gerrymandering/'+state+dataType.capitalize()+'.zip'
    districtsFile = downloadBlockAssign(state)
    #print(districtsFile)
    numOfDistricts = getNumOfDistricts(state)
    #print(numOfDistricts)
    graph,shpName = getGraphFromInfo(state,dataType,zipFilename)
    waterDis = waterDistricts(shpName)
    pops = getPopOfBlock(state)
    graph = subgraph(graph,list(Counter(list(graph.keys()))-Counter(waterDis)))
    graphFile = '/Users/samflorin/Documents/Gerrymandering/%s%sGraph.txt'%(state,dataType)
    districts = doDistricting(list(graph.keys()),districtsFile,len(list(graph.keys())[0]))
    #print(len(districts))
    #print(districts[0])
    for i in districts[0]:
        disNei = []
        neighbors = graph[i]
        print('neighbors:',neighbors)
        for j in neighbors:
            val = getDistrict(j,districts)
            print('val:',val)
            if val!=0:
                disNei.append(val)
        if len(disNei)!=0:
            actual = round(sum(disNei)/len(disNei))
        else:
            actual = 0
        print(actual)
        districts[0].remove(i)
        districts[actual].append(i)
    districts = ['blank for indexing']+[list(districts[i]) for i in range(1,len(districts))]
    for i in range(len(districts)):
        print(len(districts[i]))
        print(type(districts[i]))
    print('districts:',len(districts))
    print(numOfDistricts)
    dictionary = {}
    for i in range(1,numOfDistricts+1):
      d = districts[i]
      print(len(d))
      for j in districts[i]:
        dictionary[j]=i
    ratios = {}
    centers = {}
    matG = matrixBFS(graph)
    for district in range(1,numOfDistricts+1):
        print('On district: '+ str(district))
        parts = partitionsOf(graph,subgraph(graph,districts[district]))
        disGraph = [subgraph(graph,districts[district])]
        '''lens = [len(i) for i in parts]
        big=[]
        single = []
        for i in parts:
            if len(i)!=1:
                big.append(i)
            else:
                single.append(i)
        print(big,single)
        if len(big)>1:
            return 'uh oh, divided'
        all = []
        for i in big:
            all+=i
        '''
        saveDistrict = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'.shp'
        writeShapeFile(list(subgraph(graph,districts[district]).keys()),shpName,saveDistrict)
        zipStuff(saveDistrict[:-4],saveDistrict[:-4]+'.zip')
        for i in range(len(disGraph)):
            partSave = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'Chunk%s.shp'%(i)
            writeShapeFile(list(disGraph[i].keys()),saveDistrict,partSave)
            zipStuff(partSave[:-4],partSave[:-4]+'.zip')
        if typeOfConvexness.lower()=='hole':
            things = []
            for i in parts:
                things +=list(lineMinHole(graph,i).keys())
            #filled = lineMinHole(graph,disGraph)
        elif typeOfConvexness.lower()=='line':
            things = []
            for i in parts:
                newjunk = list(lineMin(graph,i).keys())
                things+=newjunk
                '''ChunkiedSave = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'Chunkied%s.shp'%(disGraph.index(i))
                writeShapeFile(newjunk,saveDistrict,ChunkiedSave)
                zipStuff(ChunkiedSave[:-4],ChunkiedSave[:-4]+'.zip')
'''
            #filled = lineMin(graph,disGraph)
        else:
            return 'invalid convexness type'
        things = singlify(things)
        filled = subgraph(graph,things)
        print('convexified')
        outputFile = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+typeOfConvexness.capitalize()+'.shp'
        writeShapeFile(list(filled.keys()),shpName,outputFile)
        zipStuff(outputFile[:-4],outputFile[:-4]+'.zip')
        print('written and zipped')
        newPops = {}
        l = len(list(filled.keys())[0])
        for i in list(pops.keys()):
            j = i[:l]
            if j not in list(newPops.keys()):
                newPops[j]=pops[i]
            else:
                k=newPops[j]
                k+=pops[i]
                newPops[j]=k
        s1 = 0
        for b in list(filled.keys()):
            if b in list(newPops.keys()):
                s1+=newPops[b]
        #sum([newPops[b] for b in list(filled.keys())])
        s2 = 0
        #sum([newPops[b] for b in districts[district]])
        for b in districts[district]:
            if b in list(newPops.keys()):
                s2+=newPops[b]

        ratios[district]=s2/s1
        print(s2/s1)
        ShpToJSON(outputFile,outputFile[:-3]+'json')
        centers[district] = centerAndBound(outputFile[:-3]+'json')[0]

        postGithub(outputFile[:-3]+'json')
        print('posted district ' + str(district))
    with open('/Users/samflorin/Documents/Gerrymandering/%s%s%sRatios.txt'%(state,dataType.capitalize(),typeOfConvexness.capitalize()),'w') as f:
        f.write(dumps(ratios))
    f.close()
    with open('/Users/samflorin/Documents/Gerrymandering/%s%s%sCenters.txt'%(state,dataType.capitalize(),typeOfConvexness.capitalize()),'w') as f:
        for i in list(centers.values()):
            f.write(str(i))
            f.write('\n')
    f.close()
    return ratios

def postGithub(file):
    print(file)
    key = '3249142c8eea245e50ab30756882bcde65d5fc84'
    g = Github(key)
    repo = g.get_user().get_repo('json-storage')
    file_list = [file]
    file_names = [file.split('/')[-1]]
    commit_message = 'python update 2'
    master_ref = repo.get_git_ref('heads/master')
    master_sha = master_ref.object.sha
    base_tree = repo.get_git_tree(master_sha)
    element_list = list()
    for i, entry in enumerate(file_list):
        with open(entry,'r') as input_file:
            data = input_file.read()
        if entry.endswith('.png'):
            data = base64.b64encode(data)
        element = InputGitTreeElement(file_names[i], '100644', 'blob', data)
        element_list.append(element)
    tree = repo.create_git_tree(element_list, base_tree)
    parent = repo.get_git_commit(master_sha)
    commit = repo.create_git_commit(commit_message, tree, [parent])
    master_ref.edit(commit.sha)
def getParts(state):
  datatype = 'tabblock'
  zipFilename = '/Users/samflorin/Documents/Gerrymandering/'+state+datatype.capitalize()+'.zip'

  districtsFile = downloadBlockAssign(state)
  print(districtsFile)
  numOfDistricts = getNumOfDistricts(state)
  print(numOfDistricts)
  graph,shpName = getGraphFromInfo(state,datatype,zipFilename)
  waterDis = waterDistricts(shpName)
  graph = subgraph(graph,list(Counter(list(graph.keys()))-Counter(waterDis)))
  districts = doDistricting(list(graph.keys()),districtsFile,len(list(graph.keys())[0]))
  #print(districts)
  districts = ['blank for indexing']+[list(districts[i]) for i in range(1,len(districts))]
  for i in range(len(districts)):
      print(len(districts[i]))
      print(type(districts[i]))

  dictionary = {}

  for i in range(1,numOfDistricts+1):
    for j in districts[i]:
      dictionary[j]=i
  partDict = {}
  for x in range(1,len(districts)):
    parts = partitionsOf(graph,subgraph(graph,districts[x]))
    partDict[x]=parts
    with open('/Users/samflorin/Documents/Gerrymandering/%sparts.txt'%(state),'w') as f:
        f.write(dumps(partDict))
    f.close()

def getFirstLine(state,dataType,typeOfConvexness):
    zipFilename = '/Users/samflorin/Documents/Gerrymandering/'+state+dataType.capitalize()+'.zip'
    districtsFile = downloadBlockAssign(state)
    #print(districtsFile)
    numOfDistricts = getNumOfDistricts(state)
    #print(numOfDistricts)
    graph,shpName = getGraphFromInfo(state,dataType,zipFilename)
    waterDis = waterDistricts(shpName)
    pops = getPopOfBlock(state)
    graph = subgraph(graph,list(Counter(list(graph.keys()))-Counter(waterDis)))
    graphFile = '/Users/samflorin/Documents/Gerrymandering/%s%sGraph.txt'%(state,dataType)
    districts = doDistricting(list(graph.keys()),districtsFile,len(list(graph.keys())[0]))
    #print(len(districts))
    #print(districts[0])
    for i in districts[0]:
        disNei = []
        neighbors = graph[i]
        print('neighbors:',neighbors)
        for j in neighbors:
            val = getDistrict(j,districts)
            print('val:',val)
            if val!=0:
                disNei.append(val)
        if len(disNei)!=0:
            actual = round(sum(disNei)/len(disNei))
        else:
            actual = 0
        print(actual)
        districts[0].remove(i)
        districts[actual].append(i)
    districts = ['blank for indexing']+[list(districts[i]) for i in range(1,len(districts))]
    for i in range(len(districts)):
        print(len(districts[i]))
        print(type(districts[i]))
    print('districts:',len(districts))
    print(numOfDistricts)
    dictionary = {}
    for i in range(1,numOfDistricts+1):
      d = districts[i]
      print(len(d))
      for j in districts[i]:
        dictionary[j]=i
    ratios = {}
    centers = {}
    matG = matrixBFS(graph)
    for district in range(1,numOfDistricts+1):
        print('On district: '+ str(district))
        parts = partitionsOf(graph,subgraph(graph,districts[district]))
        if len(parts)!=1:
            disGraph = connectChunks(parts,graph,subgraph(graph,districts[district]),matG)
        else:
            disGraph = [subgraph(graph,districts[district])]

        lens = [len(i) for i in parts]
        big=[]
        single = []
        for i in parts:
            if len(i)!=1:
                big.append(i)
            else:
                single.append(i)
        print(big,single)
        if len(big)>1:
            return 'uh oh, divided'
        all = []
        for i in big:
            all+=i

        saveDistrict = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'.shp'
        writeShapeFile(list(subgraph(graph,districts[district]).keys()),shpName,saveDistrict)
        zipStuff(saveDistrict[:-4],saveDistrict[:-4]+'.zip')
        for i in range(len(disGraph)):
            partSave = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'Chunk%s.shp'%(i)
            writeShapeFile(list(disGraph[i].keys()),saveDistrict,partSave)
            zipStuff(partSave[:-4],partSave[:-4]+'.zip')
        if typeOfConvexness.lower()=='hole':
            things = []
            for i in disGraph:
                things +=list(lineMinHole(graph,i).keys())
            #filled = lineMinHole(graph,disGraph)
        elif typeOfConvexness.lower()=='line':
            newLine = bestLine(graph,subgraph(graph,districts[district]),matrixBFS(graph),matrixBFS(subgraph(graph,districts[district])))[0]
            filled = addVertices(graph,subgraph(graph,districts[district]),newLine)
            #filled = lineMin(graph,disGraph)
        else:
            return 'invalid convexness type'
        #things = singlify(things)
        #filled = subgraph(graph,things)
        print('convexified')
        outputFile = '/Users/samflorin/Documents/Gerrymandering/'+state+str(district)+dataType.capitalize()+'FirstLine.shp'
        writeShapeFile(list(filled.keys()),shpName,outputFile)
        zipStuff(outputFile[:-4],outputFile[:-4]+'.zip')
        print('written and zipped')
        newPops = {}
        l = len(list(filled.keys())[0])
        for i in list(pops.keys()):
            j = i[:l]
            if j not in list(newPops.keys()):
                newPops[j]=pops[i]
            else:
                k=newPops[j]
                k+=pops[i]
                newPops[j]=k
        s1 = 0
        for b in list(filled.keys()):
            if b in list(newPops.keys()):
                s1+=newPops[b]
        #sum([newPops[b] for b in list(filled.keys())])
        s2 = 0
        #sum([newPops[b] for b in districts[district]])
        for b in districts[district]:
            if b in list(newPops.keys()):
                s2+=newPops[b]

        ratios[district]=s2/s1
        print(s2/s1)
        ShpToJSON(outputFile,outputFile[:-3]+'json')
        centers[district] = centerAndBound(outputFile[:-3]+'json')[0]

        postGithub(outputFile[:-3]+'json')
        print('posted district ' + str(district))
    '''with open('/Users/samflorin/Documents/Gerrymandering/%s%s%sRatios.txt'%(state,dataType.capitalize(),typeOfConvexness.capitalize()),'w') as f:
        f.write(dumps(ratios))
    f.close()
    with open('/Users/samflorin/Documents/Gerrymandering/%s%s%sCenters.txt'%(state,dataType.capitalize(),typeOfConvexness.capitalize()),'w') as f:
        for i in list(centers.values()):
            f.write(str(i))
            f.write('\n')
    f.close()
    '''
    return ratios

ShpToJSON('/Users/samflorin/Documents/Gerrymandering/md3Tract.shp','/Users/samflorin/Documents/Gerrymandering/md3Tract.json')
postGithub('/Users/samflorin/Documents/Gerrymandering/md3Tract.json')
'''
m = getShape('md','bg',4,'line',True)
print(m[0])

shp_to_kml.convert(m[1],(m[1])[:-3]+'kml',3)

ShpToJSON(m[1],(m[1])[:-3]+'json')
postGithub((m[1])[:-3]+'json')
'''
#writeShapeFile(['51019050100'],'/Users/samflorin/Documents/Gerrymandering/vaTract/tl_2018_51_tract.shp','/Users/samflorin/Documents/Gerrymandering/whereyouat.shp')
#zipStuff('/Users/samflorin/Documents/Gerrymandering/vaTract/tl_2018_51_tract.shp'[:-4],'/Users/samflorin/Documents/Gerrymandering/vaTract/tl_2018_51_tract.shp'[:-3]+'zip')
#print(completeState('md','bg','line'))

#postGithub('/Users/samflorin/Documents/Gerrymandering/tabblock2010_09_pophu.shp')
#line = lineMin(graphTract,subgraph(graphTract,tractDistricts[1]))
#print(line)
#writeShapeFile(list(line.keys()),shpName,'dis1line.shp')
#zipStuff('dis1line','dis1line.zip')
'''writeShapeFile(tractDistricts[1],shpName,'BGdis1.shp')
zipStuff('BGdis1','BGdis1.zip')

g = makeNiceGraph('good.shp')
print('cool')
BGs = list(g.keys())
print('nice')
districtsG = allDistricting(maximize(allDis(BGs,'BlockAssign_ST09_CT_CD.txt',len(BGs[0]))))
print('wow')
turnToSHP('good.shp','done.shp',districtsG[1])
print('done')'''
end = timeit.default_timer()
print('that took '+str(end-start)+' seconds')
