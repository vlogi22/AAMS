from collections import deque

def createMap(n: int, m:int) -> list:
  return [[0 for _ in range(0, m)] for _ in range(0, n)]

def createMapPatternCircle(row: int, col:int, range:int, iProb:float, decay:float, matrix: list) -> list:
  directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
  n, m = len(matrix), len(matrix[0])

  visited = [[0 for _ in r] for r in matrix]
  queue = deque([(row, col, iProb, range)])
  
  while queue:
    row, col, prob, r = queue.popleft()
    if (not visited[row][col]):
      matrix[row][col] = prob
      visited[row][col] = 1

      if (r > 0):
        for dr, dc in directions:
          newRow, newCol = row + dr, col + dc
          if (0 <= newRow < n) and (0 <= newCol < m) and (not visited[newRow][newCol]):
            queue.append((newRow, newCol, round(prob*decay, 4), r-1))

  return matrix

# 1 - Map with a cluster in the center
def mapCenter(n: int, m: int) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(n//2, m//2, 100, 0.02, 0.99, map)
  map = createMapPatternCircle(n//2, m//2, 5, 0.5, 0.95, map)

  return map

# 2 - Map with a 4 clusters, one in the each corner
def map4Corner(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(n//2, m//2, 100, 0.02, 0.99, map)
  
  map = createMapPatternCircle(4, 4, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(4, m-5, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(n-5, 4, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(n-5, m-5, 4, 0.5, 0.8, map)

  return map

# 3 - Map with no pattern, all the cells have same prob
def mapNoPatt(n, m) -> list:
  map = createMap(n, m)

  map = createMapPatternCircle(n//2, m//2, 100, 0.1, 0.9999, map)

  return map

# 4 - Map with a 1 clusters in the center and a ring
def mapClusterRing(n, m) -> list:
  map = createMap(n, m)
  
  map = createMapPatternCircle(n//2, m//2, 100, 0, 0, map)
  map = createMapPatternCircle(n//2, m//2, 18, 0.2, 0.95, map)
  map = createMapPatternCircle(n//2, m//2, 12, 0, 0, map)
  map = createMapPatternCircle(n//2, m//2, 3, 0.8, 0.95, map)

  return map