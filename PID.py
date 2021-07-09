#####################################
# Codes from:
#
# https://shizenkarasuzon.hatenablog.com/entry/2018/08/27/002812
#
# Modified by Chai Jiazheng
# E-mail: chai.jiazheng.q1@dc.tohoku.ac.jp
#
# 01/07/2019
#
#######################################
PI=3.14159265359

class PID:
    def __init__(self, P=5, I=0.1, D=1.5,delta_time=0.01,target_pos=0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.targetPos = target_pos
        self.delta_time=delta_time #Smallest timestep of the simulation
        self.clear()

    def clear(self):
        self.PTerm = 0
        self.ITerm = 0
        self.DTerm = 0
        self.last_error = 0

        # Windup Guard
        self.windup_guard = 10#20.0
        self.output = 0.0

    def update(self, feedback_value):

        # Feedback value takes the target position as its reference.
        # For example, if target position is 0, then when the pole
        # tilts towards the rightside of the z-axis, its value is positive.
        # Else, its value is negative.
        #if feedback_value>PI:
        #    feedback_value=-(PI-feedback_value%PI)
        if feedback_value > 2*PI:
            feedback_value = feedback_value % PI

        error = self.targetPos - feedback_value

        delta_error = error - self.last_error
        self.PTerm = self.Kp * error
        self.ITerm += error * self.delta_time

        if (self.ITerm > self.windup_guard):
            print('windguard')
            self.ITerm = self.windup_guard
        if (self.ITerm < -self.windup_guard):
            print('windguard')
            self.ITerm = -self.windup_guard

        self.DTerm = delta_error / self.delta_time
        self.last_error = error
        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

        return self.output,feedback_value

    def setTargetPosition(self, targetPos):
        self.targetPos = targetPos