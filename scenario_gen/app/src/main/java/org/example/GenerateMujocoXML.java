package org.example;

import com.google.gson.Gson;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public class GenerateMujocoXML {

    static class XY { int x, y; }
    static class Scenario {
        Map<String,Integer> grid;   // keys: "H","W"
        List<XY> agents;
        List<XY> targets;
        List<XY> obstacles;
    }

    public static void main(String[] args) throws Exception {
        // Read scenario JSON
        Scenario s;
        if (args.length > 0) {
            String json = java.nio.file.Files.readString(java.nio.file.Path.of(args[0]), StandardCharsets.UTF_8);
            s = new Gson().fromJson(json, Scenario.class);
        } else {
            // classpath: /org/example/scenario.json
            try (InputStream in = GenerateMujocoXML.class.getResourceAsStream("/org/example/scenario.json")) {
                if (in == null) throw new IllegalStateException("scenario.json not found on classpath");
                s = new Gson().fromJson(new InputStreamReader(in, StandardCharsets.UTF_8), Scenario.class);
            }
        }

        // Mapping: grid (0..W-1, 0..H-1) -> world coords within [-10,10]x[-10,10]
        // Center each body in grid cell at z=0.5.
        final double XMIN = -10.0, XMAX = 10.0;
        final double YMIN = -10.0, YMAX = 10.0;
        final int W = s.grid.get("W");
        final int H = s.grid.get("H");
        final double cellX = (XMAX - XMIN) / W;
        final double cellY = (YMAX - YMIN) / H;
        final double z = 0.5;

        // Convert (grid x,y) -> (world x,y,z)
        java.util.function.BiFunction<Integer,Integer,double[]> toWorld = (gx, gy) -> {
            // double wx = XMAX - (gx + 0.5) * cellX;
            // double wy = YMIN + (gy + 0.5) * cellY;
            double wx = XMIN + (gx + 0.5) * cellX;     // normal X (left→right)
            double wy = YMAX - (gy + 0.5) * cellY;
            return new double[]{wx, wy, z};
        };

        // 3) Agent positions
        double[] a1 = (s.agents != null && s.agents.size() >= 1)
                ? toWorld.apply(s.agents.get(0).x, s.agents.get(0).y)
                : new double[]{0.4, -8.5, z};
        double[] a2 = (s.agents != null && s.agents.size() >= 2)
                ? toWorld.apply(s.agents.get(1).x, s.agents.get(1).y)
                : new double[]{1.5, -8.5, z};

        // 4) Build XML
        StringBuilder xml = new StringBuilder();
        xml.append("""
            <mujoco model="simple_navigation">
              <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
              <option integrator="RK4" timestep="0.01"/>

              <asset>
                <texture builtin="gradient" height="100" rgb1="0.6 0.8 1" rgb2="0.1 0.1 0.1" type="skybox" width="100"/>
                <texture name="floor_texture" type="2d" builtin="checker" width="100" height="100" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5"/>
                <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="10 10" texture="floor_texture"/>
              </asset>

              <contact>
                <exclude body1="agent_1" body2="floor"/>
                <exclude body1="agent_2" body2="floor"/>
              </contact>

              <extension>
                <plugin plugin="mujoco.pid">
                  <instance name="pid1">
                    <config key="kp" value="2"/>
                    <config key="ki" value="0"/>
                    <config key="kd" value="55"/>
                  </instance>
                </plugin>
              </extension>

              <worldbody>
                <light cutoff="100" diffuse="0.9 0.9 0.9" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
                <body name="floor" pos="0 0 0">
                  <geom name="floor_geom" type="plane" size="10 10 0.1" material="MatPlane"/>
                </body>
            """);

        // Agents
        xml.append(formatAgent("agent_1", a1));
        xml.append(formatAgent("agent_2", a2));

        // Targets (yellow boxes)
        if (s.targets != null) {
            int idx = 1;
            for (XY t : s.targets) {
                double[] p = toWorld.apply(t.x, t.y);
                xml.append(formatBodyBox("target_" + idx, "target_geom_" + idx, p,
                        "1 1 0 1", 0.5, 0.5, 0.5));
                idx++;
            }
        }

        // Obstacles (black boxes)
        if (s.obstacles != null) {
            int idx = 1;
            for (XY o : s.obstacles) {
                double[] p = toWorld.apply(o.x, o.y);
                xml.append(formatBodyBox("obstacle_" + idx, "obstacle_geom_" + idx, p,
                        "0 0 0 1", 0.5, 0.5, 0.5));
                idx++;
            }
        }

        // Arena walls
        xml.append("""
                <geom name="wall_west"  type="box" pos="-10  0  0.5" size="0.05 10   0.5" rgba="0 0 0 1" contype="1" conaffinity="1"/>
                <geom name="wall_east"  type="box" pos=" 10  0  0.5" size="0.05 10   0.5" rgba="0 0 0 1" contype="1" conaffinity="1"/>
                <geom name="wall_south" type="box" pos="  0 -10 0.5" size="10   0.05 0.5" rgba="0 0 0 1" contype="1" conaffinity="1"/>
                <geom name="wall_north" type="box" pos="  0  10 0.5" size="10   0.05 0.5" rgba="0 0 0 1" contype="1" conaffinity="1"/>
              </worldbody>

              <actuator>
                <plugin joint="agent_1_j1" plugin="mujoco.pid" instance="pid1"/>
                <plugin joint="agent_1_j2" plugin="mujoco.pid" instance="pid1"/>
                <plugin joint="agent_2_j1" plugin="mujoco.pid" instance="pid1"/>
                <plugin joint="agent_2_j2" plugin="mujoco.pid" instance="pid1"/>
              </actuator>
            </mujoco>
            """);

        File outFile = new File("generated_layout.xml");
        try (Writer w = new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8)) {
            w.write(xml.toString());
        }
        System.out.println("Wrote: " + outFile.getAbsolutePath());
    }

    private static String formatAgent(String name, double[] p) {
        // each agent has two slide joints and a yellow box geom
        return String.format(java.util.Locale.ROOT,
                """
                    <body name="%s" pos="%.6f %.6f %.6f">
                        <joint name="%s_j1" type="slide" axis="1 0 0"/>
                        <joint name="%s_j2" type="slide" axis="0 1 0"/>
                        <geom name="%s_geom" type="box" size="0.5 0.5 0.5" rgba="0 0 1 1"/>
                    </body>
                """,
                name, p[0], p[1], p[2],
                name, name,
                name.replace("agent", "agent_geom")  // keep a unique geom name
        );
    }

    private static String formatBodyBox(String bodyName, String geomName, double[] p,
                                        String rgba, double sx, double sy, double sz) {
        return String.format(java.util.Locale.ROOT,
                """
                    <body name="%s" pos="%.6f %.6f %.6f">
                        <geom name="%s" type="box" size="%.6f %.6f %.6f" rgba="%s"/>
                    </body>
                """,
                bodyName, p[0], p[1], p[2],
                geomName, sx, sy, sz, rgba
        );
    }
}
