// cd scenario_gen/app/src/main/java/org/example                       

// cd /Users/susu/Desktop/Thesis_Project/scenario_gen
// ./gradlew :app:run -PmainClass=org.example.DrawGrid

package org.example;

import com.google.gson.Gson;
import javax.swing.*;
import java.awt.*;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public class DrawGrid extends JPanel {
    static class XY { int x, y; }
    static class Scenario {
        Map<String,Integer> grid;
        List<XY> agents;
        List<XY> targets;
        List<XY> obstacles;
    }

    private final Scenario s;
    private final int cell = 30;

    public DrawGrid(Scenario s) {
        this.s = s;
        int W = s.grid.get("W");
        int H = s.grid.get("H");
        setPreferredSize(new Dimension(W*cell + 1, H*cell + 1));
        setBackground(Color.WHITE);
    }

    @Override protected void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int W = s.grid.get("W");
        int H = s.grid.get("H");

        // grid lines
        g.setColor(new Color(220,220,220));
        for (int x=0; x<=W; x++) g.drawLine(x*cell, 0, x*cell, H*cell);
        for (int y=0; y<=H; y++) g.drawLine(0, y*cell, W*cell, y*cell);

        // obstacles
        g.setColor(Color.BLACK);
        for (XY o: s.obstacles) {
            g.fillRect(o.x*cell+1, o.y*cell+1, cell-1, cell-1);
        }

        // targets (orange)
        for (int i=0; i<s.targets.size(); i++) {
            XY t = s.targets.get(i);
            int cx = t.x*cell + cell/2, cy = t.y*cell + cell/2;
            g.setColor(Color.ORANGE);
            g.fillOval(cx - cell/3, cy - cell/3, 2*cell/3, 2*cell/3);
            g.setColor(Color.BLACK);
            g.drawOval(cx - cell/3, cy - cell/3, 2*cell/3, 2*cell/3);
            String lbl = "T"+i;
            FontMetrics fm = g.getFontMetrics();
            g.drawString(lbl, cx - fm.stringWidth(lbl)/2, cy + fm.getAscent()/2 - 2);
        }

        // agents (blue)
        for (int i=0; i<s.agents.size(); i++) {
            XY a = s.agents.get(i);
            int cx = a.x*cell + cell/2, cy = a.y*cell + cell/2;
            g.setColor(new Color(0, 170, 255));
            g.fillOval(cx - cell/3, cy - cell/3, 2*cell/3, 2*cell/3);
            g.setColor(Color.BLACK);
            g.drawOval(cx - cell/3, cy - cell/3, 2*cell/3, 2*cell/3);
            String lbl = "A"+i;
            FontMetrics fm = g.getFontMetrics();
            g.drawString(lbl, cx - fm.stringWidth(lbl)/2, cy + fm.getAscent()/2 - 2);
        }
    }

    public static void main(String[] args) throws Exception {

        InputStream in = DrawGrid.class.getResourceAsStream("/org/example/scenario.json");
        if (in == null) {
            throw new IllegalStateException("scenario.json not found on classpath");
        }
        Gson gson = new Gson();
        Scenario s = gson.fromJson(new InputStreamReader(in, StandardCharsets.UTF_8), Scenario.class);


        javax.swing.SwingUtilities.invokeLater(() -> {
            javax.swing.JFrame f = new javax.swing.JFrame("SAR Scenario");
            f.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
            f.setContentPane(new DrawGrid(s));
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
            f.addKeyListener(new java.awt.event.KeyAdapter() {
                @Override public void keyPressed(java.awt.event.KeyEvent e) {
                    if (e.getKeyChar() == 's' || e.getKeyChar() == 'S') {
                        java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(
                                f.getContentPane().getWidth(),
                                f.getContentPane().getHeight(),
                                java.awt.image.BufferedImage.TYPE_INT_ARGB);
                        f.getContentPane().paint(img.getGraphics());
                        try {
                            javax.imageio.ImageIO.write(img, "png", new java.io.File("scenario.png"));
                            System.out.println("Saved to scenario.png");
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    }
                }
            });
        });
    }
}