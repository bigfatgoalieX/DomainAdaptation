import mujoco
import numpy as np
import glfw

# 创建基础环境 XML 配置
ant_xml = '''
<mujoco model="ant_with_ramp">
    <compiler angle="degree" />
    <option timestep="0.01" gravity="0 0 -9.81" />
    <worldbody>
        <!-- 地面 -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>

        <!-- 斜坡 -->
        <geom name="ramp" type="box" size="2 0.5 0.1" pos="3 0 0.1" euler="0 17 0" rgba="0.8 0.6 0.4 1"/>

        <!-- 蚂蚁 -->
        <body name="ant" pos="0 0 0.5">
            <!-- 加载蚂蚁模型 -->
            <!-- 这里需要将Ant模型具体部分插入 -->
        </body>
    </worldbody>
    <actuator>
        <!-- 添加Ant的关节控制（省略，需根据Ant模型定义补充） -->
    </actuator>
</mujoco>
'''

# 加载模型
model = mujoco.MjModel.from_xml_string(ant_xml)
data = mujoco.MjData(model)

# 初始化窗口和渲染器
glfw.init()
window = glfw.create_window(800, 600, "Ant with Ramp", None, None)
glfw.make_context_current(window)
renderer = mujoco.Renderer(model, 800, 600)

# 仿真循环
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)  # 进行一步仿真

    # 渲染画面
    renderer.update_scene(data)
    renderer.render()

    # 交换缓冲区并处理事件
    glfw.swap_buffers(window)
    glfw.poll_events()

# 关闭窗口
glfw.terminate()
