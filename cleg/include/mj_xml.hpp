#pragma once
const int xml_bytes = 2992; 
const char xml_content[2992] = R"""(<mujoco model="leg">
    <option timestep="0.00025"/>
    <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" texture="texplane" texrepeat="4 4" texuniform="true"/>
    </asset>
    <worldbody>
        <geom name="ground" pos="0 0 0" size="0 0 .25" type="plane" material="matplane"/>
        <geom name="rod" type="box" pos="0 0 0" size="0.01 0.01 0.5"/>
        <light directional="false" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
        <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 4.0" dir="0 0 -1"/>
        <body name="base" pos="0 -0.025 0.5">
            <geom type="cylinder" pos="0 0 0" size="0.02 0.01" zaxis="0 1 0" rgba="1 0 0 1"/>
            <inertial pos="0 0 0" mass="0.4715" diaginertia="0.000078 0.000078 0.000078"/>
            <joint name="slider" type="slide" axis="0 0 1" frictionloss="0.55" damping="0.55" ref="0.5"/>
            <site name="base-center" pos="0 0 0"/>
            <body name="upperleg" pos="0 -0.025 0">
                <geom type="cylinder" pos="0 0 0" size="0.014 0.01" zaxis="0 1 0" rgba="0 1 0 1"/>
                <geom name="upperleg-limb" type="capsule" pos="0 0 -0.07" size="0.005 0.07 0" rgba="0 1 0 1"/>
                <inertial pos="0 -0.057 0" mass="0.315" diaginertia="0.0010888 0.0010888 0.000001"/>
                <joint name="hip" type="hinge" axis="0 -1 0" frictionloss="0.1" damping="0.01" ref="-90" armature="2.2e-6"/>
                <site name="hip-center" pos="0 0 0"/>
                <body name="lowerleg" pos="0 0 -0.14">
                    <geom name="lowerleg-limb" type="capsule" pos="0 0 -0.07" size="0.005 0.07 0" rgba="0 0 1 1"/>
                    <inertial pos="0 -0.016 0" mass="0.034" diaginertia="0.0008888 0.0008888 0.0000005"/>
                    <joint name="knee" type="hinge" axis="0 -1 0" frictionloss="0.1" damping="0.04" ref="0" armature="1.157e-5"/>
                    <site name="knee-center" pos="0 0 0"/>
                    <site name="foot-center" pos="0 0 -0.14"/>
                </body>
            </body>
        </body>
    </worldbody>
    <sensor>
        <force name="rod-to-base" site="base-center"/>
        <force name="base-to-hip" site="hip-center"/>
        <velocimeter name="base-vel" site="base-center"/>
        <torque name="hip_tau" site="hip-center"/>
        <torque name="knee_tau" site="knee-center"/>
    </sensor>
    <contact>
        <pair name="ground-contact" geom1="ground" geom2="lowerleg-limb" friction="0.7 0.7" solref="0.015 0.18" solimp="0.01 0.95 0.001"/>
    </contact>
    <actuator>
        <motor name="torquehip" joint="hip" gear="1" ctrlrange="-1 1" ctrllimited="false"/>
    </actuator>
    <actuator>
        <motor name="torqueknee" joint="knee" gear="1" ctrlrange="-1 1" ctrllimited="false"/>
    </actuator>
</mujoco>
)""";
