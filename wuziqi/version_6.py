import numpy as np
import math
import random
import pygame as pg
import os
import time
import json
from enum import Enum
import logging

####################################################################################################################
# create the initial empty chess board in the game window
board = np.zeros((16,10), dtype=int)
print(board)
def draw_board():
    global center, sep_r, sep_th, piece_radius

    center = w_size / 2
    sep_r = int((center - pad) / (radial_span - 1))  # separation between circles
    sep_th = 2 * np.pi / angular_span  # separation between radial lines
    piece_radius = sep_r / 2 * sep_th * 0.8  # size of a chess piece

    surface = pg.display.set_mode((w_size, w_size))
    pg.display.set_caption("Gomuku (a.k.a Five-in-a-Row)")

    color_line = [153, 153, 153]
    color_board = [241, 196, 15]

    surface.fill(color_board)

    for i in range(1, radial_span):
        pg.draw.circle(surface, color_line, (center, center), sep_r * i, 3)

    for i in range(angular_span // 2):
        pg.draw.line(surface, color_line,
                     (center + (center - pad) * np.cos(sep_th * i), center + (center - pad) * np.sin(sep_th * i)),
                     (center - (center - pad) * np.cos(sep_th * i), center - (center - pad) * np.sin(sep_th * i)), 3)

    pg.display.update()

    return surface
####################################################################################################################
# translate clicking position on the window to array indices (th, r)
# pos = (x,y) is a tuple returned by pygame, telling where an event (i.e. player click) occurs on the game window
def click2index(pos):
    dist = np.sqrt((pos[0] - center) ** 2 + (pos[1] - center) ** 2)
    if dist < w_size / 2 - pad + 0.25 * sep_r:  # check if the clicked position is on the circle

        # return corresponding indices (th,r) on the rectangular grid
        return (round(np.arctan2((pos[1] - center), (pos[0] - center)) / sep_th), round(dist / sep_r))

    return False  # return False if the clicked position is outside the circle
####################################################################################################################
# Draw the stones on the board at pos = [th, r]
# r and th are the indices on the 16x10 board array (under rectangular grid representation)
# Draw a black circle at pos if color = 1, and white circle at pos if color =  -1
def draw_stone(surface, pos, color=0):
    color_black = [0, 0, 0]
    color_dark_gray = [75, 75, 75]
    color_white = [255, 255, 255]
    color_light_gray = [235, 235, 235]

    # translate (th, r) indices to xy coordinate on the game window
    x = center + pos[1] * sep_r * np.cos(pos[0] * sep_th)
    y = center + pos[1] * sep_r * np.sin(pos[0] * sep_th)

    if color == 1:
        pg.draw.circle(surface, color_black, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 0)
        pg.draw.circle(surface, color_dark_gray, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 2)

    elif color == -1:
        pg.draw.circle(surface, color_white, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 0)
        pg.draw.circle(surface, color_light_gray, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 2)

    pg.display.update()
####################################################################################################################
def print_winner(surface, winner=0):
    if winner == 2:
        msg = "Draw! So White wins"
        color = [153, 153, 153]
    elif winner == 1:
        msg = "Black wins!"
        color = [0, 0, 0]
    elif winner == -1:
        msg = 'White wins!'
        color = [255, 255, 255]
    else:
        return

    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, color)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 80)
def random_move(board, color):
    time.sleep(1)
    while True:
        indx = (np.random.randint(16), np.random.randint(10))
        if board[indx] == 0:
            return indx
####################################################################################################################
####################################################################################################################
#以下是我们的part
def check_winner(board):

    if np.all(board):
        return 2
    #情况1 角度相同
    #对于x1和x2 ,如果x1==(x2+8)%16 ,那么x1和x2是同一行 对于同一行 遍历y 看看有没有5个连在一起的列
    for x1 in range(16):
        for y in range(10 - 4):  # 遍历列，最多到第6列（因为向右要有5个连续点）
            player = board[x1, y]
            if player != 0:
                if y==0:
                    count = 1
                    k=1
                    while board[x1,y+k]==player:
                        k+=1
                        count+=1
                    k=1
                    while board[(x1+8)%16,y+k]==player:
                        k+=1
                        count+=1
                    if count>=5:
                        return player
                elif all(board[x1, y + k] == player for k in range(5)):
                    return player
    # 情况二 在同一圈上
    # 此时5个点的 y坐标相等 有没有5个连在一起的x，x1=14，x2=15，x3=0，x4=1，x5=2，也会被视作连在一起
    for y in range(1,10):
        for x1 in range(16):  # 遍历所有起始点
            player = board[x1, y]
            if player != 0:
                # 检查是否在圆周上有连续5个点
                if all(board[(x1 + k) % 16, y] == player for k in range(5)):
                    return player

    # 检查情况3：对角线方向
    for x1 in range(16):
        for y in range(10 - 4):  # 限制y范围，确保y+4不越界
            player = board[x1, y]
            if player != 0:
                count = 0
                for k in range(5):
                    if board[(x1+k)%16,(y+k)%10]==player:
                        count+=1
                if count>=5:return player
                count=0
                for k in range(5):
                    if board[(x1-k)%16,(y+k)%10]==player:
                        count+=1
                if count>=5:return player
    return 0

####################################################################################################################
#computer_move
def computer_move_1(board, color):
    # 全局变量用于评分策略
    SCORE_FIVE = 100000
    SCORE_FOUR = 10000
    SCORE_BLOCKED_FOUR = 8000
    SCORE_THREE = 1000
    SCORE_BLOCKED_THREE = 800
    SCORE_TWO = 100
    SCORE_BLOCKED_TWO = 80
    SCORE_ONE = 10
    SCORE_BLOCKED_ONE = 8

    class Strategy:
        @staticmethod
        def check_continuous_line(board, pos, color, direction):
            """检查某个方向上的连续棋子"""
            x, y = pos
            dx, dy = direction
            count = 1  # 当前位置算一个
            # 正向检查
            for i in range(1, 5):
                nx = (x + dx * i) % 16
                ny = y + dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            # 反向检查
            for i in range(1, 5):
                nx = (x - dx * i) % 16
                ny = y - dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            return count

        @staticmethod
        def find_critical_moves(board, color):
            """找出关键防守位置"""
            critical_moves = []
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            for i in range(16):
                for j in range(10):
                    if board[i, j] == 0:
                        max_threat = 0
                        # 检查己方威胁
                        for dx, dy in directions:
                            board[i, j] = color
                            threat = Strategy.check_continuous_line(board, (i, j), color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        # 检查对手威胁
                        for dx, dy in directions:
                            board[i, j] = -color
                            threat = Strategy.check_continuous_line(board, (i, j), -color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        if max_threat >= 3:  # 如果有3个或以上的连子威胁
                            critical_moves.append((i, j))

            return critical_moves

        @staticmethod
        def evaluate_position(board, pos, color):
            """评估某个位置的分数"""
            if pos[1] == 0:  # 中心点特殊处理
                score = 1000
                for i in range(16):
                    if board[i, 1] == color:
                        score += 500
                    elif board[i, 1] == -color:
                        score -= 500
                return score

            score = 0
            x, y = pos
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            # 评估每个方向
            for dx, dy in directions:
                continuous = Strategy.check_continuous_line(board, pos, color, (dx, dy))
                if continuous >= 4:
                    score += SCORE_FOUR
                elif continuous == 3:
                    score += SCORE_THREE
                elif continuous == 2:
                    score += SCORE_TWO

            # 位置权重
            if 2 <= y <= 7:  # 倾向于在中间区域下棋
                score += 10

            # 检查是否能阻止对手连子
            board[pos] = -color
            opponent_threat = 0
            for dx, dy in directions:
                opponent_threat = max(opponent_threat,
                                      Strategy.check_continuous_line(board, pos, -color, (dx, dy)))
            board[pos] = 0

            if opponent_threat >= 4:
                score += SCORE_BLOCKED_FOUR
            elif opponent_threat == 3:
                score += SCORE_BLOCKED_THREE

            return score

        @staticmethod
        def get_key_positions(board):
            """获取关键位置"""
            key_positions = []
            for i in range(16):
                for j in range(10):
                    if board[i, j] != 0:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni = (i + di) % 16
                                nj = j + dj
                                if 0 <= nj < 10 and board[ni, nj] == 0:
                                    key_positions.append((ni, nj))
            return list(set(key_positions))

    class MCTSNode:
        def __init__(self, board, parent=None, move=None, color=None):
            self.board = np.copy(board)
            self.parent = parent
            self.move = move
            self.color = color
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = self._get_sorted_moves()

        def _get_sorted_moves(self):
            """获取排序后的可能移动"""
            moves = []
            key_positions = Strategy.get_key_positions(self.board)

            # 优先考虑关键位置
            for pos in key_positions:
                if self.board[pos] == 0:
                    moves.append(pos)

            # 添加其他空位置
            for i in range(16):
                for j in range(10):
                    if self.board[i, j] == 0 and (i, j) not in moves:
                        moves.append((i, j))

            if not moves:
                return moves

            # 对moves进行评分和排序
            scored_moves = []
            for move in moves:
                score = Strategy.evaluate_position(self.board, move,
                                                   self.color if self.color else 1)
                scored_moves.append((move, score))

            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in scored_moves]

        def UCT_select_child(self, exploration=1.414):
            best_score = float('-inf')
            best_child = None

            for child in self.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploitation = child.wins / child.visits
                    exploration_term = exploration * math.sqrt(2 * math.log(self.visits) / child.visits)
                    score = exploitation + exploration_term

                    if child.move:
                        pos_score = Strategy.evaluate_position(self.board, child.move, self.color) / 10000.0
                        score += 0.1 * pos_score

                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child

    def check_immediate_win(board, color):
        """检查是否有立即获胜的位置"""
        for i in range(16):
            for j in range(10):
                if board[i, j] == 0:
                    board[i, j] = color
                    if check_winner(board) == color:
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        return None

    """比赛用的主要决策函数"""
    start_time = time.time()

    # 1. 首先检查立即获胜机会
    winning_move = check_immediate_win(board, color)
    if winning_move:
        return winning_move

    # 2. 检查对手立即获胜的威胁
    blocking_move = check_immediate_win(board, -color)
    if blocking_move:
        return blocking_move

    # 3. 检查关键防守位置
    critical_moves = Strategy.find_critical_moves(board, color)
    if critical_moves:
        # 评估每个关键位置
        best_move = None
        best_score = float('-inf')
        for move in critical_moves:
            score = Strategy.evaluate_position(board, move, color)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move:
            return best_move

    # 4. 使用增强的MCTS
    root = MCTSNode(board, color=color)
    iterations = 0

    while time.time() - start_time < 4.5:
        node = root
        board_state = np.copy(board)
        current_color = color

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.UCT_select_child()
            if node.move[1] == 0:
                board_state[:, 0] = node.color
            else:
                board_state[node.move] = node.color
            current_color = -current_color

        # Expansion
        if node.untried_moves:
            move = node.untried_moves.pop(0)  # 使用预先排序的moves
            new_board = np.copy(board_state)

            if move[1] == 0:
                new_board[:, 0] = current_color
            else:
                new_board[move] = current_color

            child = MCTSNode(new_board, parent=node, move=move, color=current_color)
            node.children.append(child)
            node = child
            current_color = -current_color

        # Simulation
        sim_board = np.copy(board_state)
        sim_color = current_color

        for _ in range(50):  # 限制模拟深度
            if check_winner(sim_board):
                break

            # 检查获胜moves
            winning_move = check_immediate_win(sim_board, sim_color)
            if winning_move:
                sim_board[winning_move] = sim_color
                break

            # 检查防守moves
            blocking_move = check_immediate_win(sim_board, -sim_color)
            if blocking_move:
                sim_board[blocking_move] = sim_color
                sim_color = -sim_color
                continue

            # 获取并评估可能的moves
            possible_moves = []
            key_positions = Strategy.get_key_positions(sim_board)

            # 优先考虑关键位置
            for pos in key_positions:
                if sim_board[pos[0], pos[1]] == 0:
                    score = Strategy.evaluate_position(sim_board, pos, sim_color)
                    possible_moves.append((pos, score))

            # 如果没有关键位置，考虑所有空位
            if not possible_moves:
                for i in range(16):
                    for j in range(10):
                        if sim_board[i, j] == 0:
                            score = Strategy.evaluate_position(sim_board, (i, j), sim_color)
                            possible_moves.append(((i, j), score))

            if not possible_moves:
                break

            # 选择最佳moves或随机moves
            possible_moves.sort(key=lambda x: x[1], reverse=True)
            if random.random() < 0.8:  # 80%概率选择最佳moves
                chosen_move = possible_moves[0][0]
            else:  # 20%概率随机选择
                chosen_move = random.choice(possible_moves)[0]

            sim_board[chosen_move] = sim_color
            sim_color = -sim_color

        # Backpropagation
        result = check_winner(sim_board)
        while node:
            node.visits += 1
            if result:
                if (result == 1 and color == 1) or (result == -1 and color == -1):
                    node.wins += 1
            node = node.parent

        iterations += 1

    # 选择最佳moves
    best_child = None
    best_score = float('-inf')

    for child in root.children:
        if child.visits > 0:
            # 综合考虑UCT分数和位置评分
            uct_score = child.wins / child.visits
            position_score = Strategy.evaluate_position(board, child.move, color) / 10000.0
            visit_bonus = math.sqrt(child.visits) / 100

            total_score = uct_score + 0.3 * position_score + 0.1 * visit_bonus

            if total_score > best_score:
                best_score = total_score
                best_child = child

    if best_child is None:
        # 如果MCTS失败，使用启发式选择
        possible_moves = [(i, j) for i in range(16) for j in range(10) if board[i, j] == 0]
        if possible_moves:
            scored_moves = [(move, Strategy.evaluate_position(board, move, color))
                            for move in possible_moves]
            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return scored_moves[0][0]

    return best_child.move
def computer_move_2(board, color):
    # 全局变量用于评分策略
    SCORE_FIVE = 100000
    SCORE_FOUR = 10000
    SCORE_BLOCKED_FOUR = 8000
    SCORE_THREE = 1000
    SCORE_BLOCKED_THREE = 800
    SCORE_TWO = 100
    SCORE_BLOCKED_TWO = 80
    SCORE_ONE = 10
    SCORE_BLOCKED_ONE = 8

    class Strategy:
        @staticmethod
        def check_continuous_line(board, pos, color, direction):
            """检查某个方向上的连续棋子"""
            x, y = pos
            dx, dy = direction
            count = 1  # 当前位置算一个
            # 正向检查
            for i in range(1, 5):
                nx = (x + dx * i) % 16
                ny = y + dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            # 反向检查
            for i in range(1, 5):
                nx = (x - dx * i) % 16
                ny = y - dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            return count

        @staticmethod
        def find_critical_moves(board, color):
            """找出关键防守位置"""
            critical_moves = []
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            for i in range(16):
                for j in range(10):
                    if board[i, j] == 0:
                        max_threat = 0
                        # 检查己方威胁
                        for dx, dy in directions:
                            board[i, j] = color
                            threat = Strategy.check_continuous_line(board, (i, j), color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        # 检查对手威胁
                        for dx, dy in directions:
                            board[i, j] = -color
                            threat = Strategy.check_continuous_line(board, (i, j), -color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        if max_threat >= 3:  # 如果有3个或以上的连子威胁
                            critical_moves.append((i, j))

            return critical_moves

        @staticmethod
        def evaluate_position(board, pos, color):
            """评估某个位置的分数"""
            if pos[1] == 0:  # 中心点特殊处理
                score = 1000
                for i in range(16):
                    if board[i, 1] == color:
                        score += 500
                    elif board[i, 1] == -color:
                        score -= 500
                return score

            score = 0
            x, y = pos
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            # 评估每个方向
            for dx, dy in directions:
                continuous = Strategy.check_continuous_line(board, pos, color, (dx, dy))
                if continuous >= 4:
                    score += SCORE_FOUR
                elif continuous == 3:
                    score += SCORE_THREE
                elif continuous == 2:
                    score += SCORE_TWO

            # 位置权重
            if 2 <= y <= 7:  # 倾向于在中间区域下棋
                score += 10

            # 检查是否能阻止对手连子
            board[pos] = -color
            opponent_threat = 0
            for dx, dy in directions:
                opponent_threat = max(opponent_threat,
                                      Strategy.check_continuous_line(board, pos, -color, (dx, dy)))
            board[pos] = 0

            if opponent_threat >= 4:
                score += SCORE_BLOCKED_FOUR
            elif opponent_threat == 3:
                score += SCORE_BLOCKED_THREE

            return score

        @staticmethod
        def get_key_positions(board,color):
            """获取关键位置"""
            key_positions = []
            for i in range(16):
                for j in range(10):
                    if board[i, j] != color:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni = (i + di) % 16
                                nj = j + dj
                                if 0 <= nj < 10 and board[ni, nj] == 0:
                                    key_positions.append((ni, nj))
            return list(set(key_positions))

    class MCTSNode:
        def __init__(self, board, parent=None, move=None, color=None):
            self.board = np.copy(board)
            self.parent = parent
            self.move = move
            self.color = color
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = self._get_sorted_moves()

        def _get_sorted_moves(self):
            """获取排序后的可能移动"""
            moves = []
            key_positions = Strategy.get_key_positions(self.board,self.color)

            # 优先考虑关键位置
            for pos in key_positions:
                if self.board[pos] == 0:
                    moves.append(pos)

            # 添加其他空位置
            for i in range(16):
                for j in range(10):
                    if self.board[i, j] == 0 and (i, j) not in moves:
                        moves.append((i, j))

            if not moves:
                return moves

            # 对moves进行评分和排序
            scored_moves = []
            for move in moves:
                score = Strategy.evaluate_position(self.board, move,
                                                   self.color if self.color else 1)
                scored_moves.append((move, score))

            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in scored_moves]

        def UCT_select_child(self, exploration=1.414):
            best_score = float('-inf')
            best_child = None

            for child in self.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploitation = child.wins / child.visits
                    exploration_term = exploration * math.sqrt(2 * math.log(self.visits) / child.visits)
                    score = exploitation + exploration_term

                    if child.move:
                        pos_score = Strategy.evaluate_position(self.board, child.move, self.color) / 10000.0
                        score += 0.1 * pos_score

                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child

    def check_immediate_win(board, color):
        """检查是否有立即获胜的位置"""
        for i in range(16):
            for j in range(10):
                if board[i, j] == 0:
                    board[i, j] = color
                    if check_winner(board) == color:
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        return None

    """比赛用的主要决策函数"""
    start_time = time.time()

    # 1. 首先检查立即获胜机会
    winning_move = check_immediate_win(board, color)
    if winning_move:
        return winning_move

    # 2. 检查对手立即获胜的威胁 有就直接堵
    blocking_move = check_immediate_win(board, -color)
    if blocking_move:
        return blocking_move

    # 3. 检查关键防守位置 有就直接返回
    critical_moves = Strategy.find_critical_moves(board, color)
    if critical_moves:
        # 评估每个关键位置
        best_move = None
        best_score = float('-inf')
        for move in critical_moves:
            score = Strategy.evaluate_position(board, move, color)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move:
            print("best move")
            return best_move

    # 4. 使用增强的MCTS
    root = MCTSNode(board, color=color)
    iterations = 0

    while time.time() - start_time < 4.5:
        node = root
        board_state = np.copy(board)
        current_color = color

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.UCT_select_child()
            if node.move[1] == 0:
                board_state[:, 0] = node.color
            else:
                board_state[node.move] = node.color
            current_color = -current_color

        # Expansion
        if node.untried_moves:
            move = node.untried_moves.pop(0)  # 使用预先排序的moves
            new_board = np.copy(board_state)

            if move[1] == 0:
                new_board[:, 0] = current_color
            else:
                new_board[move] = current_color

            child = MCTSNode(new_board, parent=node, move=move, color=current_color)
            node.children.append(child)
            node = child
            current_color = -current_color

        # Simulation
        sim_board = np.copy(board_state)
        sim_color = current_color

        for _ in range(50):  # 限制模拟深度
            if check_winner(sim_board):
                break

            # 检查获胜moves
            winning_move = check_immediate_win(sim_board, sim_color)
            if winning_move:
                sim_board[winning_move] = sim_color
                break

            # 检查防守moves
            blocking_move = check_immediate_win(sim_board, -sim_color)
            if blocking_move:
                sim_board[blocking_move] = sim_color
                sim_color = -sim_color
                continue

            # 获取并评估可能的moves
            possible_moves = []
            key_positions = Strategy.get_key_positions(sim_board,color)

            # 优先考虑关键位置
            for pos in key_positions:
                if sim_board[pos[0], pos[1]] == 0:
                    score = Strategy.evaluate_position(sim_board, pos, sim_color)
                    possible_moves.append((pos, score))

            # 如果没有关键位置，考虑所有空位
            if not possible_moves:
                for i in range(16):
                    for j in range(10):
                        if sim_board[i, j] == 0:
                            score = Strategy.evaluate_position(sim_board, (i, j), sim_color)
                            possible_moves.append(((i, j), score))

            if not possible_moves:
                break

            # 选择最佳moves或随机moves
            possible_moves.sort(key=lambda x: x[1], reverse=True)
            if random.random() < 0.8:  # 80%概率选择最佳moves
                chosen_move = possible_moves[0][0]
            else:  # 20%概率随机选择
                chosen_move = random.choice(possible_moves)[0]

            sim_board[chosen_move] = sim_color
            sim_color = -sim_color

        # Backpropagation
        result = check_winner(sim_board)
        while node:
            node.visits += 1
            if result:
                if (result == 1 and color == 1) or (result == -1 and color == -1):
                    node.wins += 1
            node = node.parent

        iterations += 1

    # 选择最佳moves
    best_child = None
    best_score = float('-inf')

    for child in root.children:
        if child.visits > 0:
            # 综合考虑UCT分数和位置评分
            uct_score = child.wins / child.visits
            position_score = Strategy.evaluate_position(board, child.move, color) / 10000.0
            visit_bonus = math.sqrt(child.visits) / 100

            total_score = uct_score + 0.3 * position_score + 0.1 * visit_bonus

            if total_score > best_score:
                best_score = total_score
                best_child = child

    if best_child is None:
        # 如果MCTS失败，使用启发式选择
        possible_moves = [(i, j) for i in range(16) for j in range(10) if board[i, j] == 0]
        if possible_moves:
            scored_moves = [(move, Strategy.evaluate_position(board, move, color))
                            for move in possible_moves]
            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return scored_moves[0][0]

    return best_child.move
def computer_move_3(board, color):
    # 全局变量用于评分策略
    SCORE_FIVE = 100000
    SCORE_FOUR = 10000
    SCORE_BLOCKED_FOUR = 8000
    SCORE_THREE = 1000
    SCORE_BLOCKED_THREE = 800
    SCORE_TWO = 100
    SCORE_BLOCKED_TWO = 80
    SCORE_ONE = 10
    SCORE_BLOCKED_ONE = 8

    class Strategy:
        @staticmethod
        def check_continuous_line(board, pos, color, direction):
            """检查某个方向上的连续棋子"""
            x, y = pos
            dx, dy = direction
            count = 1  # 当前位置算一个
            # 正向检查
            for i in range(1, 5):
                nx = (x + dx * i) % 16
                ny = y + dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            # 反向检查
            for i in range(1, 5):
                nx = (x - dx * i) % 16
                ny = y - dy * i
                if 0 <= ny < 10 and board[nx, ny] == color:
                    count += 1
                else:
                    break
            return count

        @staticmethod
        def find_critical_moves(board, color):
            """找出关键防守位置"""
            critical_moves = []
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            for i in range(16):
                for j in range(10):
                    if board[i, j] == 0:
                        max_threat = 0
                        # 检查己方威胁
                        for dx, dy in directions:
                            board[i, j] = color
                            threat = Strategy.check_continuous_line(board, (i, j), color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        # 检查对手威胁
                        for dx, dy in directions:
                            board[i, j] = -color
                            threat = Strategy.check_continuous_line(board, (i, j), -color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        if max_threat >= 3:  # 如果有3个或以上的连子威胁
                            critical_moves.append((i, j))

            return critical_moves

        @staticmethod
        def find_critical_moves_fs(board, color):
            """找出关键防守位置"""
            critical_moves = []
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            for i in range(16):
                for j in range(10):
                    if board[i, j] == 0:
                        max_threat = 0
                        # 检查对手威胁
                        for dx, dy in directions:
                            board[i, j] = -color
                            threat = Strategy.check_continuous_line(board, (i, j), -color, (dx, dy))
                            board[i, j] = 0
                            max_threat = max(max_threat, threat)

                        if max_threat >= 3:  # 如果有3个或以上的连子威胁
                            critical_moves.append((i, j))

            return critical_moves
        @staticmethod
        def evaluate_position(board, pos, color):
            """评估某个位置的分数"""
            if pos[1] == 0:  # 中心点特殊处理
                score = 1000
                for i in range(16):
                    if board[i, 1] == color:
                        score += 500
                    elif board[i, 1] == -color:
                        score -= 500
                return score

            score = 0
            x, y = pos
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

            # 评估每个方向
            for dx, dy in directions:
                continuous = Strategy.check_continuous_line(board, pos, color, (dx, dy))
                if continuous >= 4:
                    score += SCORE_FOUR
                elif continuous == 3:
                    score += SCORE_THREE
                elif continuous == 2:
                    score += SCORE_TWO

            # 位置权重
            if 2 <= y <= 7:  # 倾向于在中间区域下棋
                score += 10

            # 检查是否能阻止对手连子
            board[pos] = -color
            opponent_threat = 0
            for dx, dy in directions:
                opponent_threat = max(opponent_threat,
                                      Strategy.check_continuous_line(board, pos, -color, (dx, dy)))
            board[pos] = 0

            if opponent_threat >= 4:
                score += SCORE_BLOCKED_FOUR
            elif opponent_threat == 3:
                score += SCORE_BLOCKED_THREE

            return score

        @staticmethod
        def get_key_positions(board, color):
            """获取关键位置"""
            key_positions = []
            for i in range(16):
                for j in range(10):
                    if board[i, j] != color:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni = (i + di) % 16
                                nj = j + dj
                                if 0 <= nj < 10 and board[ni, nj] == 0:
                                    key_positions.append((ni, nj))
            return list(set(key_positions))

    class MCTSNode:
        def __init__(self, board, parent=None, move=None, color=None):
            self.board = np.copy(board)
            self.parent = parent
            self.move = move
            self.color = color
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = self._get_sorted_moves()

        def _get_sorted_moves(self):
            """获取排序后的可能移动"""
            moves = []
            key_positions = Strategy.get_key_positions(self.board, self.color)

            # 优先考虑关键位置
            for pos in key_positions:
                if self.board[pos] == 0:
                    moves.append(pos)

            # 添加其他空位置
            for i in range(16):
                for j in range(10):
                    if self.board[i, j] == 0 and (i, j) not in moves:
                        moves.append((i, j))

            if not moves:
                return moves

            # 对moves进行评分和排序
            scored_moves = []
            for move in moves:
                score = Strategy.evaluate_position(self.board, move,
                                                   self.color if self.color else 1)
                scored_moves.append((move, score))

            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in scored_moves]

        def UCT_select_child(self, exploration=1.414):
            best_score = float('-inf')
            best_child = None

            for child in self.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploitation = child.wins / child.visits
                    exploration_term = exploration * math.sqrt(2 * math.log(self.visits) / child.visits)
                    score = exploitation + exploration_term

                    if child.move:
                        pos_score = Strategy.evaluate_position(self.board, child.move, self.color) / 10000.0
                        score += 0.1 * pos_score

                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child

    def check_immediate_win(board, color):
        """检查是否有立即获胜的位置"""
        for i in range(16):
            for j in range(10):
                if board[i, j] == 0:
                    board[i, j] = color
                    if check_winner(board) == color:
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        return None

    """比赛用的主要决策函数"""
    start_time = time.time()

    # 1. 首先检查立即获胜机会
    winning_move = check_immediate_win(board, color)
    if winning_move:
        return winning_move

    # 2. 检查对手立即获胜的威胁 有就直接堵
    blocking_move = check_immediate_win(board, -color)
    if blocking_move:
        return blocking_move

    # 3. 检查关键防守位置 有就直接返回
    critical_moves = Strategy.find_critical_moves_fs(board, color)
    if critical_moves:
        # 评估每个关键位置
        best_move = None
        best_score = float('-inf')
        for move in critical_moves:
            score = Strategy.evaluate_position(board, move, color)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move:
            print("best move")
            return best_move

    # 4. 使用增强的MCTS
    root = MCTSNode(board, color=color)
    iterations = 0

    while time.time() - start_time < 4.5:
        node = root
        board_state = np.copy(board)
        current_color = color

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.UCT_select_child()
            if node.move[1] == 0:
                board_state[:, 0] = node.color
            else:
                board_state[node.move] = node.color
            current_color = -current_color

        # Expansion
        if node.untried_moves:
            move = node.untried_moves.pop(0)  # 使用预先排序的moves
            new_board = np.copy(board_state)

            if move[1] == 0:
                new_board[:, 0] = current_color
            else:
                new_board[move] = current_color

            child = MCTSNode(new_board, parent=node, move=move, color=current_color)
            node.children.append(child)
            node = child
            current_color = -current_color

        # Simulation
        sim_board = np.copy(board_state)
        sim_color = current_color

        for _ in range(50):  # 限制模拟深度
            if check_winner(sim_board):
                break

            # 检查获胜moves
            winning_move = check_immediate_win(sim_board, sim_color)
            if winning_move:
                sim_board[winning_move] = sim_color
                break

            # 检查防守moves
            blocking_move = check_immediate_win(sim_board, -sim_color)
            if blocking_move:
                sim_board[blocking_move] = sim_color
                sim_color = -sim_color
                continue

            # 获取并评估可能的moves
            possible_moves = []
            key_positions = Strategy.get_key_positions(sim_board, color)

            # 优先考虑关键位置
            for pos in key_positions:
                if sim_board[pos[0], pos[1]] == 0:
                    score = Strategy.evaluate_position(sim_board, pos, sim_color)
                    possible_moves.append((pos, score))

            # 如果没有关键位置，考虑所有空位
            if not possible_moves:
                for i in range(16):
                    for j in range(10):
                        if sim_board[i, j] == 0:
                            score = Strategy.evaluate_position(sim_board, (i, j), sim_color)
                            possible_moves.append(((i, j), score))

            if not possible_moves:
                break

            # 选择最佳moves或随机moves
            possible_moves.sort(key=lambda x: x[1], reverse=True)
            if random.random() < 0.8:  # 80%概率选择最佳moves
                chosen_move = possible_moves[0][0]
            else:  # 20%概率随机选择
                chosen_move = random.choice(possible_moves)[0]

            sim_board[chosen_move] = sim_color
            sim_color = -sim_color

        # Backpropagation
        result = check_winner(sim_board)
        while node:
            node.visits += 1
            if result:
                if (result == 1 and color == 1) or (result == -1 and color == -1):
                    node.wins += 1
            node = node.parent

        iterations += 1

    # 选择最佳moves
    best_child = None
    best_score = float('-inf')

    for child in root.children:
        if child.visits > 0:
            # 综合考虑UCT分数和位置评分
            uct_score = child.wins / child.visits
            position_score = Strategy.evaluate_position(board, child.move, color) / 10000.0
            visit_bonus = math.sqrt(child.visits) / 100

            total_score = uct_score + 0.3 * position_score + 0.1 * visit_bonus

            if total_score > best_score:
                best_score = total_score
                best_child = child

    if best_child is None:
        # 如果MCTS失败，使用启发式选择
        possible_moves = [(i, j) for i in range(16) for j in range(10) if board[i, j] == 0]
        if possible_moves:
            scored_moves = [(move, Strategy.evaluate_position(board, move, color))
                            for move in possible_moves]
            scored_moves.sort(key=lambda x: x[1], reverse=True)
            return scored_moves[0][0]

    return best_child.move
def computer_move_4(board, color):
    # 全局常量
    BOARD_SIZE_ANGULAR = 16
    BOARD_SIZE_RADIAL = 10
    WINDOW_SIZE = 720
    PADDING = 36

    # 评分常量
    SCORE_FIVE = 100000
    SCORE_FOUR = 10000
    SCORE_BLOCKED_FOUR = 8000
    SCORE_THREE = 1000
    SCORE_BLOCKED_THREE = 800
    SCORE_TWO = 100
    SCORE_BLOCKED_TWO = 80
    SCORE_ONE = 10
    SCORE_BLOCKED_ONE = 8
    class OpeningStyle(Enum):
        AGGRESSIVE = 'aggressive'
        DEFENSIVE = 'defensive'
        CENTER_CONTROL = 'center_control'
        BALANCED = 'balanced'
        FLEXIBLE = 'flexible'

    class OpeningBook:
        def __init__(self):
            self.openings = {
                # 攻击性开局
                'aggressive': [
                    # 三角进攻
                    ([(8, 3), (7, 4), (9, 4)], 100),
                    # 斜线突破
                    ([(8, 3), (8, 4), (7, 4)], 95),
                    # 双线进攻
                    ([(8, 3), (7, 3), (9, 3)], 90),
                    # 包围策略
                    ([(8, 2), (7, 3), (9, 3), (8, 4)], 85),
                    # 箭头阵型
                    ([(8, 2), (7, 3), (9, 3), (8, 1)], 80),
                ],

                # 防守性开局
                'defensive': [
                    # 铁壁防守
                    ([(8, 2), (7, 3), (9, 3), (8, 4)], 100),
                    # 盾形防守
                    ([(8, 2), (7, 2), (9, 2), (8, 3)], 95),
                    # 双翼布局
                    ([(8, 2), (6, 2), (10, 2)], 90),
                    # 防守三角
                    ([(8, 2), (7, 3), (9, 3)], 85),
                    # 稳健布局
                    ([(8, 2), (8, 3), (8, 4)], 80),
                ],

                # 中心控制开局
                'center_control': [
                    # 中心点占领
                    ([(8, 0)], 100),
                    # 中心三角
                    ([(8, 1), (7, 2), (9, 2)], 95),
                    # 中心十字
                    ([(8, 1), (7, 1), (9, 1), (8, 2)], 90),
                    # 中心方块
                    ([(8, 1), (7, 1), (7, 2), (8, 2)], 85),
                    # 中心梯形
                    ([(8, 1), (6, 2), (10, 2), (8, 3)], 80),
                ],

                # 灵活开局
                'flexible': [
                    # 双线灵活
                    ([(8, 2), (7, 3), (9, 4)], 100),
                    # 斜线布局
                    ([(8, 2), (9, 3), (7, 4)], 95),
                    # 跳跃布局
                    ([(8, 2), (6, 3), (10, 3)], 90),
                    # Z字形布局
                    ([(8, 2), (7, 3), (9, 3), (10, 4)], 85),
                    # 不规则布局
                    ([(8, 2), (6, 3), (9, 3), (7, 4)], 80),
                ],

                # 平衡开局
                'balanced': [
                    # 稳健平衡
                    ([(8, 2), (8, 3), (7, 3), (9, 3)], 100),
                    # 双向布局
                    ([(8, 2), (7, 3), (9, 3), (8, 4)], 95),
                    # 均衡三角
                    ([(8, 2), (6, 3), (10, 3), (8, 4)], 90),
                    # 方块布局
                    ([(8, 2), (7, 2), (7, 3), (8, 3)], 85),
                    # 五点布局
                    ([(8, 2), (6, 3), (10, 3), (7, 4), (9, 4)], 80),
                ]
            }

            # 开局应对策略
            self.responses = {
                # 对应不同开局的应对方式
                (8, 3): [(7, 2), (9, 2), (8, 4)],  # 对中心进攻的应对
                (8, 2): [(8, 4), (7, 3), (9, 3)],  # 对保守开局的应对
                (8, 0): [(8, 2), (7, 1), (9, 1)],  # 对中心点开局的应对
            }

            # 开局风格权重
            self.style_weights = {
                OpeningStyle.AGGRESSIVE: {
                    'win_rate': 0.3,
                    'position_control': 0.2,
                    'flexibility': 0.1,
                    'defensive_value': 0.1,
                    'attack_potential': 0.3
                },
                OpeningStyle.DEFENSIVE: {
                    'win_rate': 0.2,
                    'position_control': 0.3,
                    'flexibility': 0.1,
                    'defensive_value': 0.3,
                    'attack_potential': 0.1
                },
                OpeningStyle.CENTER_CONTROL: {
                    'win_rate': 0.25,
                    'position_control': 0.35,
                    'flexibility': 0.15,
                    'defensive_value': 0.15,
                    'attack_potential': 0.1
                },
                OpeningStyle.FLEXIBLE: {
                    'win_rate': 0.2,
                    'position_control': 0.2,
                    'flexibility': 0.4,
                    'defensive_value': 0.1,
                    'attack_potential': 0.1
                },
                OpeningStyle.BALANCED: {
                    'win_rate': 0.25,
                    'position_control': 0.25,
                    'flexibility': 0.2,
                    'defensive_value': 0.15,
                    'attack_potential': 0.15
                }
            }

        def get_opening_move(self, board, color, style='balanced'):
            """获取开局移动"""
            move_count = np.count_nonzero(board)

            if move_count > 6:
                return None

            # 如果是第一步，随机选择一个开局策略
            if move_count == 0:
                if random.random() < 0.3:  # 30%概率使用中心点开局
                    return (8, 0)
                else:
                    return self._get_first_move(style)

            # 检查是否需要应对对手的开局
            last_move = self._get_last_move(board, -color)
            if last_move in self.responses:
                response_moves = self.responses[last_move]
                for move in response_moves:
                    if board[move] == 0:
                        return move

            # 根据风格选择开局序列
            available_styles = [style]
            if style == 'balanced':
                available_styles = list(OpeningStyle)

            for opening_style in available_styles:
                for moves, score in self.openings[opening_style.value]:
                    if self._matches_sequence(board, moves, color):
                        next_move = self._get_next_move(board, moves)
                        if next_move:
                            return next_move

            return None

        def _get_first_move(self, style):
            """获取第一步移动"""
            if style in self.openings:
                # 从该风格的开局中随机选择一个
                opening_sequence = random.choice(self.openings[style])
                return opening_sequence[0][0]
            return (8, 2)  # 默认第一步

        def _get_last_move(self, board, color):
            """获取对手最后一步移动"""
            for i in range(BOARD_SIZE_ANGULAR):
                for j in range(BOARD_SIZE_RADIAL):
                    if board[i, j] == color:
                        return (i, j)
            return None

        def _matches_sequence(self, board, sequence, color):
            """检查局面是否匹配开局序列"""
            move_count = 0
            for x, y in sequence:
                if board[x, y] != 0:
                    move_count += 1

            if move_count == 0:
                return True

            for i in range(move_count):
                x, y = sequence[i]
                expected_color = 1 if i % 2 == 0 else -1
                if board[x, y] != expected_color * color:
                    return False

            return True

        def _get_next_move(self, board, sequence):
            """获取序列中的下一个移动"""
            for x, y in sequence:
                if board[x, y] == 0:
                    return (x, y)
            return None

        def evaluate_opening(self, board, sequence, color, style):
            """评估开局序列"""
            score = 0
            weights = self.style_weights[OpeningStyle(style)]

            # 评估位置控制
            control_score = self._evaluate_position_control(board, sequence)
            score += control_score * weights['position_control']

            # 评估灵活性
            flexibility_score = self._evaluate_flexibility(board, sequence)
            score += flexibility_score * weights['flexibility']

            # 评估防守价值
            defensive_score = self._evaluate_defensive_value(board, sequence)
            score += defensive_score * weights['defensive_value']

            # 评估进攻潜力
            attack_score = self._evaluate_attack_potential(board, sequence)
            score += attack_score * weights['attack_potential']

            return score

        def _evaluate_position_control(self, board, sequence):
            """评估位置控制"""
            score = 0
            center_distance = np.zeros((BOARD_SIZE_ANGULAR, BOARD_SIZE_RADIAL))

            # 计算到中心的距离
            for i in range(BOARD_SIZE_ANGULAR):
                for j in range(BOARD_SIZE_RADIAL):
                    center_distance[i, j] = abs(i - 8) + abs(j - 5)

            for x, y in sequence:
                score += (10 - center_distance[x, y]) * 10

            return score

        def _evaluate_flexibility(self, board, sequence):
            """评估灵活性"""
            score = 0
            moves = set(sequence)

            for x, y in sequence:
                # 检查周围空位
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = (x + dx) % BOARD_SIZE_ANGULAR
                        ny = y + dy
                        if 0 <= ny < BOARD_SIZE_RADIAL:
                            if (nx, ny) not in moves:
                                score += 5

            return score

        def _evaluate_defensive_value(self, board, sequence):
            """评估防守价值"""
            score = 0
            for x, y in sequence:
                if y <= 3:  # 靠近己方的位置
                    score += 20
                if 2 <= y <= 7:  # 中间区域
                    score += 10

            return score

        def _evaluate_attack_potential(self, board, sequence):
            """评估进攻潜力"""
            score = 0
            for x, y in sequence:
                if y >= 6:  # 靠近对方的位置
                    score += 20
                if 2 <= y <= 7:  # 中间区域
                    score += 10

            return score

    class PositionEvaluator:
        """位置评估器"""

        def __init__(self):
            self.pattern_cache = {}
            self.direction_weights = {
                (0, 1): 1.0,  # 垂直
                (1, 0): 1.0,  # 水平
                (1, 1): 0.9,  # 主对角线
                (1, -1): 0.9,  # 副对角线
            }

        def evaluate_pattern(self, line, color):
            """评估棋型"""
            pattern_str = ''.join(map(str, line))
            cache_key = (pattern_str, color)

            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]

            score = 0
            length = len(line)

            # 连五
            for i in range(length - 4):
                if all(line[j] == color for j in range(i, i + 5)):
                    score += SCORE_FIVE
                    break

            # 活四
            for i in range(length - 5):
                if (line[i] == 0 and
                        all(line[j] == color for j in range(i + 1, i + 5)) and
                        line[i + 5] == 0):
                    score += SCORE_FOUR

            # 冲四
            for i in range(length - 4):
                if all(line[j] == color for j in range(i, i + 4)) and (
                        (i > 0 and line[i - 1] == 0) or
                        (i + 4 < length and line[i + 4] == 0)):
                    score += SCORE_BLOCKED_FOUR

            # 活三
            for i in range(length - 4):
                if (line[i] == 0 and
                        line[i + 1] == color and
                        line[i + 2] == color and
                        line[i + 3] == color and
                        line[i + 4] == 0):
                    score += SCORE_THREE

            # 眠三
            for i in range(length - 3):
                if (all(line[j] == color for j in range(i, i + 3)) and
                        (i == 0 or line[i - 1] != color) and
                        (i + 3 >= length or line[i + 3] != color)):
                    score += SCORE_BLOCKED_THREE

            # 活二
            for i in range(length - 3):
                if (line[i] == 0 and
                        line[i + 1] == color and
                        line[i + 2] == color and
                        line[i + 3] == 0):
                    score += SCORE_TWO

            self.pattern_cache[cache_key] = score
            return score

        def evaluate_territory(self, board, color):
            """评估领地控制"""
            score = 0
            # 位置权重矩阵
            weights = np.array([
                [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
                [2, 3, 4, 5, 5, 5, 5, 4, 3, 2],
                [3, 4, 5, 6, 6, 6, 6, 5, 4, 3],
                [4, 5, 6, 7, 7, 7, 7, 6, 5, 4],
                [4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
                [4, 5, 6, 7, 8, 8, 7, 6, 5, 4],
                [4, 5, 6, 7, 7, 7, 7, 6, 5, 4],
                [3, 4, 5, 6, 6, 6, 6, 5, 4, 3],
                [2, 3, 4, 5, 5, 5, 5, 4, 3, 2],
                [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
            ])

            for i in range(BOARD_SIZE_ANGULAR):
                for j in range(BOARD_SIZE_RADIAL):
                    if board[i, j] == color:
                        score += weights[j, min(i, 9)] * 10
                    elif board[i, j] == -color:
                        score -= weights[j, min(i, 9)] * 10

            return score

        def evaluate_connectivity(self, board, pos, color):
            """评估连接性"""
            score = 0
            x, y = pos

            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                connected = 0
                # 正向检查
                for i in range(1, 4):
                    nx = (x + dx * i) % BOARD_SIZE_ANGULAR
                    ny = y + dy * i
                    if 0 <= ny < BOARD_SIZE_RADIAL:
                        if board[nx, ny] == color:
                            connected += 1
                        else:
                            break

                # 反向检查
                for i in range(1, 4):
                    nx = (x - dx * i) % BOARD_SIZE_ANGULAR
                    ny = y - dy * i
                    if 0 <= ny < BOARD_SIZE_RADIAL:
                        if board[nx, ny] == color:
                            connected += 1
                        else:
                            break

                score += connected * 50

            return score

        def evaluate_mobility(self, board, pos):
            """评估位置机动性"""
            x, y = pos
            score = 0
            empty_count = 0

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % BOARD_SIZE_ANGULAR
                    ny = y + dy
                    if 0 <= ny < BOARD_SIZE_RADIAL and board[nx, ny] == 0:
                        empty_count += 1

            return empty_count * 30

    class Strategy:
        def __init__(self):
            self.evaluator = PositionEvaluator()
            self.opening_book = OpeningBook()

        def evaluate_position(self, board, pos, color):
            """综合评估位置"""
            if pos[1] == 0:  # 中心列特殊处理
                return self._evaluate_center_column(board, pos, color)

            score = 0
            x, y = pos

            # 基础棋型评估
            pattern_score = self._evaluate_patterns(board, pos, color)
            score += pattern_score

            # 领地控制评估
            territory_score = self.evaluator.evaluate_territory(board, color)
            score += territory_score * 0.5

            # 连接性评估
            connectivity_score = self.evaluator.evaluate_connectivity(board, pos, color)
            score += connectivity_score * 0.3

            # 机动性评估
            mobility_score = self.evaluator.evaluate_mobility(board, pos)
            score += mobility_score * 0.2

            # 威胁评估
            threat_score = self._evaluate_threats(board, pos, color)
            score += threat_score * 0.8

            return score

        def _evaluate_center_column(self, board, pos, color):
            """评估中心列位置"""
            score = 1500
            for i in range(BOARD_SIZE_ANGULAR):
                if board[i, 1] == color:
                    score += 750
                elif board[i, 1] == -color:
                    score -= 750
            return score

        def _evaluate_patterns(self, board, pos, color):
            """评估所有方向的棋型"""
            score = 0
            x, y = pos

            for (dx, dy), weight in self.evaluator.direction_weights.items():
                line = []
                for i in range(-5, 6):
                    nx = (x + dx * i) % BOARD_SIZE_ANGULAR
                    ny = y + dy * i
                    if 0 <= ny < BOARD_SIZE_RADIAL:
                        line.append(board[nx, ny])
                    else:
                        line.append(2)  # 边界标记
                score += self.evaluator.evaluate_pattern(line, color) * weight

            return score

        def _evaluate_threats(self, board, pos, color):
            """评估威胁"""
            score = 0
            x, y = pos
            board[x, y] = -color  # 临时模拟对手在此位置下子

            # 检查对手潜在威胁
            for dx, dy in self.evaluator.direction_weights.keys():
                line = []
                for i in range(-5, 6):
                    nx = (x + dx * i) % BOARD_SIZE_ANGULAR
                    ny = y + dy * i
                    if 0 <= ny < BOARD_SIZE_RADIAL:
                        line.append(board[nx, ny])
                    else:
                        line.append(2)
                score += self.evaluator.evaluate_pattern(line, -color)

            board[x, y] = 0  # 恢复位置
            return score

        def get_key_positions(self, board):
            """获取关键位置"""
            key_positions = []
            for i in range(BOARD_SIZE_ANGULAR):
                for j in range(BOARD_SIZE_RADIAL):
                    if board[i, j] != 0:
                        # 检查周围空位
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni = (i + di) % BOARD_SIZE_ANGULAR
                                nj = j + dj
                                if 0 <= nj < BOARD_SIZE_RADIAL and board[ni, nj] == 0:
                                    key_positions.append((ni, nj))

            return list(set(key_positions))

        def find_critical_moves(self, board, color):
            """找出关键防守位置"""
            critical_moves = []
            key_positions = self.get_key_positions(board)

            for pos in key_positions:
                # 评估进攻价值
                attack_score = self.evaluate_position(board, pos, color)
                # 评估防守价值
                defense_score = self.evaluate_position(board, pos, -color)

                if attack_score >= SCORE_THREE or defense_score >= SCORE_THREE:
                    critical_moves.append((pos, max(attack_score, defense_score)))

            # 按分数排序
            critical_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in critical_moves]

    class MCTSNode:
        def __init__(self, board, parent=None, move=None, color=None):
            self.board = np.copy(board)
            self.parent = parent
            self.move = move
            self.color = color
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = None
            self.strategy = Strategy()

        def get_untried_moves(self):
            """获取未尝试的移动"""
            if self.untried_moves is None:
                self.untried_moves = self._get_sorted_moves()
            return self.untried_moves

        def _get_sorted_moves(self):
            """获取排序后的可能移动"""
            moves = []

            # 检查立即获胜和防守移动
            winning_move = check_immediate_win(self.board, self.color)
            if winning_move:
                return [winning_move]

            blocking_move = check_immediate_win(self.board, -self.color)
            if blocking_move:
                return [blocking_move]

            # 获取关键位置
            critical_moves = self.strategy.find_critical_moves(self.board, self.color)
            if critical_moves:
                return critical_moves[:10]  # 只保留前10个最关键的位置

            # 获取并评估所有可能的移动
            key_positions = self.strategy.get_key_positions(self.board)
            for pos in key_positions:
                if self.board[pos] == 0:
                    score = self.strategy.evaluate_position(self.board, pos, self.color)
                    moves.append((pos, score))

            # 如果关键位置不够，添加其他空位置
            if len(moves) < 10:
                for i in range(BOARD_SIZE_ANGULAR):
                    for j in range(BOARD_SIZE_RADIAL):
                        if self.board[i, j] == 0 and (i, j) not in key_positions:
                            score = self.strategy.evaluate_position(self.board, (i, j), self.color)
                            moves.append(((i, j), score))

            moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, _ in moves[:20]]  # 返回前20个最佳移动

        def UCT_select_child(self, exploration=1.414):
            """使用UCT公式选择子节点"""
            best_score = float('-inf')
            best_children = []
            total_visits = sum(child.visits for child in self.children)

            for child in self.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    # UCT公式 = 胜率 + 探索因子 + 位置评分 + 访问率奖励
                    win_rate = child.wins / child.visits
                    exploration = exploration * math.sqrt(math.log(total_visits) / child.visits)
                    position_score = self.strategy.evaluate_position(
                        self.board, child.move, self.color) / 10000.0
                    visit_ratio = math.sqrt(child.visits / total_visits)

                    score = (win_rate * 0.4 +  # 胜率权重
                             exploration * 0.3 +  # 探索权重
                             position_score * 0.2 +  # 位置评分权重
                             visit_ratio * 0.1)  # 访问率权重

                if score > best_score:
                    best_score = score
                    best_children = [child]
                elif score == best_score:
                    best_children.append(child)

            return random.choice(best_children)

    def simulate_game(board, color, strategy, max_depth=50):
        """模拟游戏"""
        current_board = np.copy(board)
        current_color = color
        depth = 0

        while depth < max_depth:
            winner = check_winner(current_board)
            if winner:
                return winner

            move = get_simulation_move(current_board, current_color, strategy)
            if move is None:
                break

            if move[1] == 0:
                current_board[:, 0] = current_color
            else:
                current_board[move] = current_color

            current_color = -current_color
            depth += 1

        # 评估最终局面
        final_score = strategy.evaluate_position(current_board, None, color)
        return 1 if final_score > 0 else -1

    def get_simulation_move(board, color, strategy):
        """获取模拟移动"""
        # 检查获胜机会
        winning_move = check_immediate_win(board, color)
        if winning_move:
            return winning_move

        # 检查防守需求
        blocking_move = check_immediate_win(board, -color)
        if blocking_move:
            return blocking_move

        # 获取可能的移动
        possible_moves = []
        key_positions = strategy.get_key_positions(board)

        # 评估关键位置
        for pos in key_positions:
            if board[pos] == 0:
                score = strategy.evaluate_position(board, pos, color)
                possible_moves.append((pos, score))

        # 如果没有关键位置，考虑所有空位
        if not possible_moves:
            for i in range(BOARD_SIZE_ANGULAR):
                for j in range(BOARD_SIZE_RADIAL):
                    if board[i, j] == 0:
                        score = strategy.evaluate_position(board, (i, j), color)
                        possible_moves.append(((i, j), score))

        if not possible_moves:
            return None

        # 选择移动
        possible_moves.sort(key=lambda x: x[1], reverse=True)
        if random.random() < 0.8:  # 80%选择最佳移动
            return possible_moves[0][0]
        else:  # 20%随机选择前三个移动
            return random.choice(possible_moves[:3])[0]

    def select_best_move(root, board, color, strategy):
        """选择最佳移动"""
        best_child = None
        best_score = float('-inf')

        for child in root.children:
            if child.visits == 0:
                continue

            # 综合评分
            win_rate = child.wins / child.visits
            position_score = strategy.evaluate_position(board, child.move, color) / 10000.0
            visit_ratio = math.sqrt(child.visits / root.visits)

            total_score = (win_rate * 0.6 +
                           position_score * 0.3 +
                           visit_ratio * 0.1)

            if total_score > best_score:
                best_score = total_score
                best_child = child

        return best_child.move if best_child else fallback_move(board, color, strategy)

    def fallback_move(board, color, strategy):
        """获取备选移动"""
        # 在出错时使用简单的启发式方法选择移动
        moves = []
        for i in range(BOARD_SIZE_ANGULAR):
            for j in range(BOARD_SIZE_RADIAL):
                if board[i, j] == 0:
                    score = strategy.evaluate_position(board, (i, j), color)
                    moves.append(((i, j), score))

        if moves:
            moves.sort(key=lambda x: x[1], reverse=True)
            return moves[0][0]
        return None
    def check_immediate_win(board, color):
        """检查是否有立即获胜的位置"""
        for i in range(BOARD_SIZE_ANGULAR):
            for j in range(BOARD_SIZE_RADIAL):
                if board[i, j] == 0:
                    board[i, j] = color
                    if check_winner(board) == color:
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        return None
    """AI决策函数"""
    try:
        start_time = time.time()
        strategy = Strategy()

        # 1. 使用开局库
        opening_move = strategy.opening_book.get_opening_move(board, color)
        if opening_move:
            return opening_move

        # 2. 检查立即获胜机会
        winning_move = check_immediate_win(board, color)
        if winning_move:
            return winning_move

        # 3. 检查必须防守的威胁
        blocking_move = check_immediate_win(board, -color)
        if blocking_move:
            return blocking_move

        # 4. 使用MCTS搜索
        root = MCTSNode(board, color=color)
        iterations = 0

        while time.time() - start_time < 4.5:  # 限制思考时间
            # Selection
            node = root
            current_board = np.copy(board)
            current_color = color

            while node.get_untried_moves() == [] and node.children:
                node = node.UCT_select_child()
                if node.move[1] == 0:
                    current_board[:, 0] = current_color
                else:
                    current_board[node.move] = current_color
                current_color = -current_color

            # Expansion
            if node.get_untried_moves():
                move = node.get_untried_moves().pop(0)
                new_board = np.copy(current_board)
                if move[1] == 0:
                    new_board[:, 0] = current_color
                else:
                    new_board[move] = current_color

                child = MCTSNode(
                    new_board,
                    parent=node,
                    move=move,
                    color=-current_color
                )
                node.children.append(child)
                node = child
                current_color = -current_color

            # Simulation
            sim_board = np.copy(current_board)
            sim_result = simulate_game(sim_board, current_color, strategy)

            # Backpropagation
            while node:
                node.visits += 1
                if sim_result == color:
                    node.wins += 1
                node = node.parent

            iterations += 1

        # 选择最佳移动
        return select_best_move(root, board, color, strategy)

    except Exception as e:
        logging.error(f"Error in computer_move: {e}")
        return fallback_move(board, color, strategy)

def main(player_is_black=True):
    global w_size, pad, radial_span, angular_span
    w_size = 720  # window size
    pad = 36  # padding size
    radial_span = 10
    angular_span = 16

    pg.init()
    surface = draw_board()

    board = np.zeros((angular_span, radial_span), dtype=int)
    running = True
    gameover = False

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                break

        if not gameover:
            # 人类玩家回合
            start_time = time.time()
            move_made = False
            timeout = 10  # 10秒超时
            while time.time() - start_time < timeout and not move_made:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                        break
                    if event.type == pg.MOUSEBUTTONDOWN:
                        indx = click2index(event.pos)
                        if indx and board[indx] == 0:
                            print("black", indx)
                            if indx[1] == 0:
                                for k in range(16):
                                    board[k, indx[1]] = 1
                            board[indx] = 1
                            draw_stone(surface, indx, 1)
                            move_made = True
                            break
                        else:
                            print("This position is already occupied. Please try again.")

                if not move_made:
                    pg.display.flip()
                    time.sleep(0.1)

            #超时了人还没下
            if not move_made and not gameover:
                print("Time's up! Human player skipped.")

            # 检查人类玩家回合后的游戏状态
            gameover = check_winner(board)
            print(board)
            if gameover:
                print_winner(surface, gameover)
                print(board)
                continue

            if np.all(board):
                gameover = True
                continue

            # 电脑回合
            if not gameover:
                indx = computer_move_4(board, -1)
                print("white", indx)
                print(board)
                if board[indx] == 0:
                    if indx[1] == 0:
                        for k in range(16):
                            board[k, indx[1]] = -1
                    board[indx] = -1
                    draw_stone(surface, indx, -1)
                    gameover = check_winner(board)
                    if gameover:
                        print_winner(surface, gameover)
                        print(board)
                else:
                    print("This position is already occupied. Computer's turn is skipped.")

            if np.all(board):
                gameover = True

        pg.display.flip()  # 更新显示

    pg.quit()
def battle(computer1_move, computer2_move):
    global w_size, pad, radial_span, angular_span
    w_size = 720
    pad = 36
    radial_span = 10
    angular_span = 16

    pg.init()
    surface = draw_board()

    board = np.zeros((angular_span, radial_span), dtype=int)
    running = True
    gameover = False

    while running:

        for event in pg.event.get():

            if event.type == pg.QUIT:
                running = False

        if not gameover:
            indx = computer1_move(board, 1)  # First group is assigned to be black
            print("black", indx)
            if board[indx] == 0:
                if indx[1] == 0:
                    for k in range(16):
                        board[k, indx[1]] = 1
                board[indx] = 1
                draw_stone(surface, indx, 1)
                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    print(board)
            else:
                print("This position is already occupied. Therefore your turn is skipped.")

        if np.all(board):
            gameover = True
        if not gameover:
            indx = computer2_move(board, -1)
            print("white", indx)
            if board[indx] == 0:
                if indx[1] == 0:
                    for k in range(16):
                        board[k, indx[1]] = -1
                board[indx] = -1
                draw_stone(surface, indx, -1)
                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    print(board)
            else:
                print("This position is already occupied. Computer's turn is skipped.")


        if np.all(board):
            gameover = True

    pg.quit()
if __name__ == '__main__':
    #battle(computer_move_3,computer_move_4)
    main()



