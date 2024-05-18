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

def map1(n: int, m: int) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(n//2, m//2, 30, 0.1, 0.9, map)
  map = createMapPatternCircle(n//2, m//2, 10, 0.5, 0.8, map)

  return map

def map2(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(n//2, m//2, 15, 0.02, 0.99, map)
  
  map = createMapPatternCircle(3, 3, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(3, m-3, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(n-3, 3, 4, 0.5, 0.8, map)
  map = createMapPatternCircle(n-3, m-3, 4, 0.5, 0.8, map)

  return map

def map3(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(10, 10, 100, 0.05, 0.9, map)
  
  map = createMapPatternCircle(25, 3, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(0, 22, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(46, 0, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(40, 25, 5, 0.5, 0.8, map)

  return map

def map4(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(20, 0, 100, 0.05, 0.9, map)
  
  map = createMapPatternCircle(5, 5, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(0, 25, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(25, 0, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(40, 35, 5, 0.5, 0.8, map)

  return map

def map5(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(20, 30, 100, 0.05, 0.9, map)
  
  map = createMapPatternCircle(3, 44, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(0, 5, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(12, 0, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(2, 35, 5, 0.5, 0.8, map)

  return map

def map6(n, m) -> list:
  map = createMap(n, m)
  map = createMapPatternCircle(40, 10, 100, 0.05, 0.9, map)
  
  map = createMapPatternCircle(2, 33, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(42, 33, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(10, 4, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(22, 10, 5, 0.5, 0.8, map)

  return map

# 7 - Map with a circle in the center, centered food source
def map7(n: int, m: int) -> list:
    map = createMap(n, m)
    center_row, center_col = n // 2, m // 2
    radius = min(n, m) // 2
    initial_prob = 0.9
    decay = 0.8
    map = createMapPatternCircle(center_row, center_col, radius, initial_prob, decay, map)
    return map

