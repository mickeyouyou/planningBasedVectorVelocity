import signal
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

LANE_WIDTH = 0.4  # meter
VEHICLE_WIDTH = 0.21  # meter


class Vector2d():

    def __init__(self, x, y):
        self.deltaX = x
        self.deltaY = y
        self.length = -1
        self.direction = [0, 0]
        self.vector2d_share()

    def vector2d_share(self):
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()):
            deltaX, deltaY = self.deltaX, self.deltaY
            self.deltaX = deltaY[0] - deltaX[0]
            self.deltaY = deltaY[1] - deltaX[1]

        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2)
        self.direction = [self.deltaX / self.length,
                          self.deltaY / self.length] if self.length > 0 else None

    def __add__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.vector2d_share()
        return vec

    def __sub__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.vector2d_share()
        return vec

    def __mul__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.vector2d_share()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(self.deltaX, self.deltaY, self.length,
                                                                             self.direction)


class APF():

    def __init__(self, start: (), goal: (), obstacles: [], rr: float,
                 max_iters: int, goal_threshold: float, is_plot=False):
        """
        :param start: 起点
        :param goal: 终点
        :param obstacles: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param max_iters: 最大迭代次数
        :param goal_threshold: 离目标点小于此值即认为到达目标点
        :param is_plot: 是否绘图
        """
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.rr = rr  # 规避场作用范围
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.is_plot = is_plot
        self.delta_t = 0.001
        self.w = 10

    def attractive(self):
        """
        目标速度计算
        """
        # 方向由机器人指向目标点
        denominator = math.sqrt((self.goal.deltaX - self.current_pos.deltaX)
                                ** 2 + (self.goal.deltaY - self.current_pos.deltaY)**2)

        rtx = (self.goal.deltaX - self.current_pos.deltaX) / denominator
        rty = (self.goal.deltaY - self.current_pos.deltaY) / denominator
        att = Vector2d(rtx, rty) * self.w
        return att

    def repulsion(self):
        """
        规避速度计算
        """
        R_i = 0.02
        deltaR = 0.1

        alpha = 0.04
        beta = 0.25
        W_P = beta*self.w
        L = deltaR  # / math.sqrt(alpha/beta-1)
        E = 1

        matrix = np.array([[0, -1], [1, 0]])

        R_1 = np.array([[math.cos(math.pi/2), -math.sin(math.pi/2)],
                        [math.sin(math.pi/2), math.cos(math.pi/2)]])

        R_2 = np.array([[math.cos(- math.pi/2), -math.sin(-math.pi/2)],
                        [math.sin(-math.pi/2), math.cos(-math.pi/2)]])

        rep = Vector2d(0, 0)
        for obstacle in self.obstacles:
            v_pi = Vector2d(0, 0)
            v_g_l = Vector2d(0, 0)
            # 规避速度向量
            # TODO 选择哪个距离？
            t_vec = self.current_pos - obstacle
            distance2obstacle = math.sqrt((self.goal.deltaX - self.current_pos.deltaX)
                                          ** 2 + (self.goal.deltaY - self.current_pos.deltaY) ** 2)
            print('t_vec length:{}, distance2obstacle:{}'.format(
                t_vec.length, distance2obstacle))
            # 超出障碍物规避速度影响范围
            if t_vec.length > R_i + deltaR:
                pass
            elif t_vec.length <= R_i:
                # 方向由障碍物垂直指向圆环外围
                v_pi = obstacle * W_P/(t_vec.length/R_i)**2
            else:
                v_pi = obstacle * W_P/(1 + ((t_vec.length - R_i)/L)**2)

            if v_pi.length > 0:
                print('v_pi:{}'.format(v_pi))
                # 导引速度向量
                matrix_vpi = np.array([[v_pi.deltaX, v_pi.deltaY]])
                # 规避速度旋转pi/2
                # v_g = R_1 * matrix_vpi
                v_g = np.array([[-v_pi.deltaY], [v_pi.deltaX]])
                print('dot production result v_g:{}'.format(v_g))

                # 规避速度旋转-pi/2
                # v_gg = R_2 * matrix_vpi
                v_gg = np.array([[v_pi.deltaY], [-v_pi.deltaX]])
                print('dot production result v_gg:{}'.format(v_gg))

                goal = np.array([[self.goal.deltaX], [self.goal.deltaY]])
                # m = np.dot(v_g, goal)
                m = -v_pi.deltaY * self.goal.deltaX + v_pi.deltaX * self.goal.deltaY
                print('m :{}'.format(m))

                # n = np.dot(v_gg, goal)
                n = v_pi.deltaY * self.goal.deltaX - v_pi.deltaX * self.goal.deltaY
                print('n:{}'.format(n))

                # # v_g 与goal 成锐角，说明靠近指向goal
                if m > 0:
                    v_g_l = Vector2d(-v_pi.deltaX, -v_pi.deltaY) * E

                # # v_g 与goal 成锐角，说明靠近指向goal
                if n > 0:
                    v_g_l = Vector2d(
                        v_pi.deltaY, -v_pi.deltaX) * E * v_pi.length

                print('v_g_l:{}'.format(v_g_l))

            # 叠加规避速度
            # rep += v_pi
            # 叠加导引速度
            rep += v_g_l

        return rep

    def road(self):
        road_rep = Vector2d(0, 0)
        # 道路边界作为斥力加入计算
        kl = 0.5
        kr = 0.5  # 道路左右边界斥力系数
        # print('i is :{}'.format(i))
        # TODO 撒点排查当前点
        left_bound = Vector2d(self.current_pos.deltaX, LANE_WIDTH / 2)
        right_bound = Vector2d(self.current_pos.deltaX, - LANE_WIDTH / 2)

        lb_vector = left_bound - self.current_pos
        rb_vector = self.current_pos - right_bound
        print('left_vector:{}'.format(lb_vector))
        print('right_vector:{}'.format(rb_vector))

        f_l = Vector2d(lb_vector.direction[0], lb_vector.direction[1]) * kl * \
            (1/(lb_vector.length - VEHICLE_WIDTH/2)*(1/self.current_pos.deltaY**2))

        f_r = Vector2d(rb_vector.direction[0], rb_vector.direction[1]) * kr * \
            (1/(rb_vector.length - VEHICLE_WIDTH/2)*(1/self.current_pos.deltaY**2))

        print('new point:{}'.format(self.current_pos))
        road_rep += f_l
        road_rep += f_r
        return road_rep

    def path_plan(self):
        while (self.iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threashold):
            attractive = self.attractive()
            print('attractive : {}'.format(attractive))
            repulsion = self.repulsion()
            print('repulsion: {}'.format(repulsion))
            v_vec = attractive + repulsion

            print('combined velocity:{}'.format(v_vec))
            # 下一个点的选择
            self.current_pos = Vector2d(
                self.current_pos.deltaX + v_vec.deltaX * self.delta_t, self.current_pos.deltaY + v_vec.deltaY * self.delta_t)
            print('current_position:{}'.format(self.current_pos))
            # exit(0)
            self.iters += 1
            self.path.append(
                [self.current_pos.deltaX, self.current_pos.deltaY])
            if self.is_plot:
                plt.plot(self.current_pos.deltaX,
                         self.current_pos.deltaY, '.r')
                plt.pause(self.delta_t)
        if (self.current_pos - self.goal).length <= self.goal_threashold:
            self.is_path_plan_success = True

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        X = np.arange(1.0, 3.0, 0.1)
        Y = np.arange(1.3, 1.7, 0.1)
        X, Y = np.meshgrid(X, Y)
        # R = np.sqrt(X**2 + Y**2)
        # Z = np.arange(1000, -1000)
        Z = np.sqrt(X**2 + Y**2)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()


def handler(signum, frame):
    print('You choose to stop me.')


signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


if __name__ == '__main__':
    rr = 0.10
    max_iters, goal_threashold = 1000, 0.03

    # 设置、绘制起点终点
    start, goal = (1, 1.5), (2.0, 1.5)

    # 障碍物设置及绘制
    obs = [[1.25, 1.45]]
    print('obstacles: {0}'.format(obs))

    is_plot = True
    if is_plot:
        fig = plt.figure(figsize=(10, 6))
        subplot = fig.add_subplot(111)
        subplot.set_xlabel('X-distance: m')
        subplot.set_ylabel('Y-distance: m')
        subplot.plot(start[0], start[1], '*y')
        subplot.plot(goal[0], goal[1], '*r')

    # 车道绘制
    stright_path_length = 2
    path_width = 0.4
    subplot.add_patch(
        patches.Rectangle(
            (1, 1.3),   # (x,y)
            stright_path_length,
            path_width,
        )
    )
    subplot.add_patch(
        patches.Rectangle(
            (1, 2.3),   # (x,y)
            stright_path_length,
            path_width,
        )
    )

    left_wedge = patches.Wedge((1, 2), .7, 90, -90, width=path_width)
    right_wedge = patches.Wedge((3, 2), .7, -90, 90, width=path_width)
    subplot.add_patch(left_wedge)
    subplot.add_patch(right_wedge)

    subplot.grid(True, linestyle="-.", color="r", linewidth=0.1)

    if is_plot:
        for OB in obs:
            circle = Circle(xy=(OB[0], OB[1]), radius=rr,
                            alpha=0.3, color='coral')
            subplot.add_patch(circle)
            subplot.plot(OB[0], OB[1], 'xk', color='coral')
    # plt.show()

    apf = APF(start, goal, obs, rr, max_iters, goal_threashold, is_plot)
    apf.path_plan()
    # apf.plot()
    # plt.show()
    if apf.is_path_plan_success:
        path = apf.path
        # path_ = []
        # i = int(step_size_ / step_size)
        # while (i < len(path)):
        #     path_.append(path[i])
        #     i += int(step_size_ / step_size)

        # if path_[-1] != path[-1]:
        #     path_.append(path[-1])
        # print('planed path points:{}'.format(path_))
        print('path plan success')
        if is_plot:
            px, py = [K[0] for K in path], [K[1] for K in path]
            subplot.plot(px, py, '>b')
            plt.show()
    else:
        print('path plan failed')
