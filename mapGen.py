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

def map1() -> list:
  map = createMap(50, 50)
  map = createMapPatternCircle(25, 25, 50, 0.1, 0.9, map)
  map = createMapPatternCircle(25, 25, 10, 0.5, 0.8, map)

  return map

def map2() -> list:
  map = createMap(50, 50)
  map = createMapPatternCircle(25, 25, 50, 0.1, 0.9, map)
  
  map = createMapPatternCircle(0, 0, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(0, 49, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(49, 0, 5, 0.5, 0.8, map)
  map = createMapPatternCircle(49, 49, 5, 0.5, 0.8, map)

  return map


