package com.racas.forzamotorsport;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.res.ColorStateList;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Native options screen shown before SDL starts the game. */
public final class LauncherActivity extends Activity {
    private static final int WHITE = Color.WHITE;
    private static final int REQUEST_PICK_DRIVER = 4102;
    private final GameImporter importer = new GameImporter(this);
    private View settingsCard;
    private LinearLayout settingsPanel;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        buildScreen();
    }

    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (!focused || android.os.Build.VERSION.SDK_INT < 30) return;
        WindowInsetsController c = getWindow().getInsetsController();
        if (c != null) {
            c.hide(WindowInsets.Type.systemBars());
            c.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
        }
    }

    private void buildScreen() {
        FrameLayout root = new FrameLayout(this);
        ImageView background = new ImageView(this);
        background.setImageResource(R.drawable.launcher_background);
        background.setScaleType(ImageView.ScaleType.CENTER_CROP);
        root.addView(background, new FrameLayout.LayoutParams(-1, -1));
        View shade = new View(this);
        shade.setBackground(new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT,
                new int[] {0x78000000, 0x26000000, 0x00000000}));
        root.addView(shade, new FrameLayout.LayoutParams(-1, -1));

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.HORIZONTAL);
        content.setGravity(Gravity.CENTER_VERTICAL);
        content.setPadding(dp(18), dp(5), dp(18), dp(5));
        root.addView(content, new FrameLayout.LayoutParams(-1, -1));

        LinearLayout menu = new LinearLayout(this);
        menu.setOrientation(LinearLayout.VERTICAL);
        menu.setGravity(Gravity.CENTER_VERTICAL);
        menu.setTranslationY(-dp(20));
        LinearLayout.LayoutParams menuParams = new LinearLayout.LayoutParams(0, -2, 0.88f);
        menuParams.rightMargin = dp(16);
        content.addView(menu, menuParams);
        menu.addView(button(LauncherIcon.PLAY, "JOGAR", true, () -> startActivity(new Intent(this, ForzaActivity.class))));
        menu.addView(button(LauncherIcon.SETTINGS, "CONFIGURAÇÕES", false, this::toggleSettings));
        menu.addView(button(LauncherIcon.DISC, "SELECIONAR ISO", false, this::pickIso));
        menu.addView(button(LauncherIcon.EXIT, "SAIR", false, this::finish));
        TextView version = label("MOTORFORZA  ·  ANDROID", 9, 0xE6FFFFFF, true);
        version.setGravity(Gravity.CENTER);
        version.setBackground(outlined(0xAA101216, 0x88FFFFFF, 7));
        LinearLayout.LayoutParams vp = new LinearLayout.LayoutParams(-1, dp(26));
        vp.topMargin = dp(14);
        menu.addView(version, vp);

        ScrollView scroller = new ScrollView(this);
        // Keep the settings card close to the compact reference layout instead of
        // stretching it from edge to edge on tall displays. It remains scrollable
        // on smaller screens.
        scroller.setFillViewport(false);
        scroller.setBackground(outlined(0xD9161A20, 0xB8FFFFFF, 10));
        scroller.setClipToPadding(false);
        android.util.DisplayMetrics metrics = getResources().getDisplayMetrics();
        // In landscape some Android versions report the unrotated display metrics.
        // Use the short edge so the card follows the reference proportions on phones.
        int screenHeight = Math.min(metrics.widthPixels, metrics.heightPixels);
        int availableHeight = Math.max(dp(220), screenHeight * 76 / 100);
        int panelHeight = Math.min(dp(340), Math.min(availableHeight, Math.max(dp(1), screenHeight - dp(20))));
        settingsCard = scroller;
        scroller.setTranslationY(-dp(24));
        LinearLayout.LayoutParams panelParams = new LinearLayout.LayoutParams(0, panelHeight, 0.82f);
        content.addView(scroller, panelParams);
        View spacer = new View(this);
        content.addView(spacer, new LinearLayout.LayoutParams(0, 1, 1.15f));
        settingsPanel = new LinearLayout(this);
        settingsPanel.setOrientation(LinearLayout.VERTICAL);
        settingsPanel.setPadding(dp(11), dp(1), dp(11), dp(1));
        scroller.addView(settingsPanel);
        addSettings();
        setContentView(root);
    }

    private View button(int icon, String title, boolean primary, Runnable action) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(14), 0, dp(12), 0);
        row.setBackground(outlined(primary ? 0xD91B2027 : 0xC90C1016, 0xA8FFFFFF, 18));
        row.setElevation(dp(2));
        row.setClickable(true);
        row.setFocusable(true);
        row.setContentDescription(title);
        row.setDefaultFocusHighlightEnabled(false);
        row.setOnClickListener(view -> action.run());
        row.setOnFocusChangeListener((view, focused) ->
                row.setBackground(outlined(primary ? 0xD91B2027 : 0xC90C1016,
                        0xA8FFFFFF, 18)));

        LauncherIcon glyph = new LauncherIcon(icon);
        row.addView(glyph, new LinearLayout.LayoutParams(dp(40), dp(40)));
        TextView titleView = label(title, 16, WHITE, true);
        titleView.setSingleLine(true);
        titleView.setGravity(Gravity.CENTER_VERTICAL);
        titleView.setPadding(dp(10), 0, 0, 0);
        row.addView(titleView, new LinearLayout.LayoutParams(0, -1, 1));

        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, dp(58));
        p.bottomMargin = dp(11);
        row.setLayoutParams(p);
        return row;
    }

    /** Small source-drawn icons keep the launcher consistent across Android font sets. */
    private final class LauncherIcon extends View {
        static final int PLAY = 0, SETTINGS = 1, DISC = 2, EXIT = 3;
        private final int kind;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Path path = new Path();

        LauncherIcon(int kind) {
            super(LauncherActivity.this);
            this.kind = kind;
            setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        }

        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float side = Math.min(getWidth(), getHeight());
            canvas.save();
            canvas.translate((getWidth() - side) * .5f, (getHeight() - side) * .5f);
            canvas.scale(side / 32f, side / 32f);
            paint.setColor(Color.WHITE);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.2f);
            paint.setStrokeCap(Paint.Cap.ROUND);
            paint.setStrokeJoin(Paint.Join.ROUND);

            switch (kind) {
                case PLAY:
                    canvas.drawCircle(16, 16, 12, paint);
                    path.reset(); path.moveTo(13, 10); path.lineTo(22, 16); path.lineTo(13, 22); path.close();
                    paint.setStyle(Paint.Style.FILL); canvas.drawPath(path, paint);
                    break;
                case SETTINGS:
                    // Eight-point gear outline with a clear center, matching the reference's cog symbol.
                    path.reset();
                    for (int i = 0; i < 32; i++) {
                        double a = (Math.PI * 2 * i / 32.0) - Math.PI / 2;
                        int phase = i % 4;
                        float radius = (phase == 0 || phase == 1) ? 12 : 9;
                        float x = 16 + (float) Math.cos(a) * radius;
                        float y = 16 + (float) Math.sin(a) * radius;
                        if (i == 0) path.moveTo(x, y); else path.lineTo(x, y);
                    }
                    path.close(); canvas.drawPath(path, paint);
                    canvas.drawCircle(16, 16, 3.5f, paint);
                    break;
                case DISC:
                    canvas.drawCircle(16, 16, 12, paint);
                    canvas.drawCircle(16, 16, 3, paint);
                    canvas.drawLine(18.2f, 13.8f, 25, 7, paint);
                    canvas.drawCircle(23.3f, 8.7f, 1.1f, paint);
                    break;
                default: // exit
                    canvas.drawRoundRect(5, 4, 18, 28, 1.5f, 1.5f, paint);
                    canvas.drawLine(14, 16, 27, 16, paint);
                    canvas.drawLine(22, 11, 27, 16, paint);
                    canvas.drawLine(22, 21, 27, 16, paint);
                    break;
            }
            canvas.restore();
        }
    }

    // A TextView subclass is not needed for layout; this method is replaced by addButton below.
    private void addSettings() {
        settingsPanel.addView(section("RENDERIZAÇÃO"));
        
        String driverName = readString("custom_driver_name", "Padrão do Sistema (Vulkan Mali)");
        TextView driverStatus = label("Driver: " + driverName, 10, WHITE, true);
        driverStatus.setPadding(dp(2), dp(2), dp(2), dp(2));
        settingsPanel.addView(driverStatus);

        LinearLayout driverBtns = new LinearLayout(this);
        driverBtns.setOrientation(LinearLayout.HORIZONTAL);

        TextView importBtn = label("IMPORTAR (.ZIP)", 9, WHITE, true);
        importBtn.setGravity(Gravity.CENTER);
        importBtn.setPadding(dp(4), dp(2), dp(4), dp(2));
        importBtn.setBackground(outlined(0x44203040, 0x885090FF, 6));
        importBtn.setOnClickListener(v -> pickDriverZip());
        driverBtns.addView(importBtn, new LinearLayout.LayoutParams(0, dp(32), 1.0f));

        View btnSpace = new View(this);
        driverBtns.addView(btnSpace, new LinearLayout.LayoutParams(dp(6), 1));

        TextView resetBtn = label("RESTAURAR", 9, 0xFFE0E0E0, true);
        resetBtn.setGravity(Gravity.CENTER);
        resetBtn.setPadding(dp(4), dp(2), dp(4), dp(2));
        resetBtn.setBackground(outlined(0x44302020, 0x88FF6060, 6));
        resetBtn.setOnClickListener(v -> resetSystemDriver());
        driverBtns.addView(resetBtn, new LinearLayout.LayoutParams(0, dp(32), 1.0f));

        LinearLayout.LayoutParams dbp = new LinearLayout.LayoutParams(-1, -2);
        dbp.topMargin = dp(2);
        dbp.bottomMargin = dp(4);
        settingsPanel.addView(driverBtns, dbp);

        settingsPanel.addView(label("Resolução Interna", 10, 0xFFE4E7EB, false));
        addSpinner(new String[] {"Nativa 480p (Xbox 1x - Mais Rapido)", "540p (50%)", "720p (67%)", "1080p (100%)"},
                new int[] {44, 50, 67, 100}, readInt("render_percent", 44), "render_percent");
        LinearLayout filterRow = new LinearLayout(this);
        filterRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView filterLabel = label("Filtro Anisotrópico", 10, 0xFFE4E7EB, false);
        filterRow.addView(filterLabel, new LinearLayout.LayoutParams(0, dp(30), 1));
        Switch aniso = new Switch(this);
        aniso.setShowText(false);
        aniso.setThumbTintList(new ColorStateList(
                new int[][] {new int[] {android.R.attr.state_checked}, new int[] {}},
                new int[] {0xFFFFFFFF, 0xFFE2E5E9}));
        aniso.setTrackTintList(new ColorStateList(
                new int[][] {new int[] {android.R.attr.state_checked}, new int[] {}},
                new int[] {0xFFFFC400, 0xFF747A82}));
        aniso.setChecked(readInt("anisotropic_filtering", 0) > 0);
        aniso.setOnCheckedChangeListener((b, checked) -> writeSetting("anisotropic_filtering", checked ? 16 : 0));
        filterRow.addView(aniso, new LinearLayout.LayoutParams(-2, dp(30)));
        settingsPanel.addView(filterRow);

        settingsPanel.addView(section("CONTROLES"));
        addValueRow("Controle de Toque", "Xbox Clássico");
        settingsPanel.addView(label("Mapeamento de Gamepad", 10, 0xFFE4E7EB, false));
        addSpinner(new String[] {"Padrão", "Trocar A / B", "Trocar analógicos"}, new int[] {0, 1, 2},
                readInt("gamepad_preset", 0), "gamepad_preset");
        settingsPanel.addView(label("Força de Feedback", 10, 0xFFE4E7EB, false));
        SeekBar feedback = new SeekBar(this);
        feedback.setMax(100);
        feedback.setProgress(readPercent("force_feedback", 100));
        feedback.setProgressTintList(ColorStateList.valueOf(0xFFF3F4F5));
        feedback.setThumbTintList(ColorStateList.valueOf(0xFFFFFFFF));
        feedback.setProgressBackgroundTintList(ColorStateList.valueOf(0xFF777C82));
        feedback.setMaxHeight(dp(4));
        feedback.setMinimumHeight(dp(24));
        feedback.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar bar, int progress, boolean fromUser) {
                if (fromUser) writeSetting("force_feedback", progress);
            }
            public void onStartTrackingTouch(SeekBar bar) {}
            public void onStopTrackingTouch(SeekBar bar) {}
        });
        settingsPanel.addView(feedback);

        settingsPanel.addView(section("SISTEMA"));
        addValueRow("Idioma", "Internacional");
        TextView update = label("Verificar atualizações", 12, WHITE, true);
        update.setGravity(Gravity.CENTER);
        update.setBackground(round(0x55333333, 7));
        update.setOnClickListener(v -> new AlertDialog.Builder(this).setTitle("Atualizações")
                .setMessage("A verificação automática de atualizações ainda não está configurada.")
                .setPositiveButton("OK", null).show());
        LinearLayout.LayoutParams up = new LinearLayout.LayoutParams(-1, dp(29));
        up.topMargin = dp(2);
        settingsPanel.addView(update, up);
    }

    private void recreateSettings() {
        if (settingsPanel != null) {
            settingsPanel.removeAllViews();
            addSettings();
        }
    }

    private void pickDriverZip() {
        try {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {
                    "application/zip", "application/x-zip-compressed", "application/octet-stream"
            });
            startActivityForResult(intent, REQUEST_PICK_DRIVER);
        } catch (Exception e) {
            Toast.makeText(this, "Não foi possível abrir o seletor de arquivos: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void resetSystemDriver() {
        File driverDir = new File(getFilesDir(), "custom_driver");
        if (driverDir.exists()) {
            File[] files = driverDir.listFiles();
            if (files != null) for (File f : files) f.delete();
            driverDir.delete();
        }
        writeSettingString("custom_driver_name", "Padrão do Sistema (Vulkan Mali)");
        writeSetting("custom_driver_enabled", 0);
        Toast.makeText(this, "Driver restaurado para o padrão do sistema (Vulkan Mali).", Toast.LENGTH_LONG).show();
        recreateSettings();
    }

    private void importDriverZip(android.net.Uri uri) {
        new Thread(() -> {
            try {
                File driverDir = new File(getFilesDir(), "custom_driver");
                if (!driverDir.exists()) driverDir.mkdirs();

                String metaJson = null;
                java.util.List<String> extractedFiles = new java.util.ArrayList<>();

                try (java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(getContentResolver().openInputStream(uri))) {
                    java.util.zip.ZipEntry entry;
                    byte[] buffer = new byte[16384];
                    while ((entry = zis.getNextEntry()) != null) {
                        String name = entry.getName();
                        if (entry.isDirectory() || name.contains("__MACOSX") || name.startsWith(".")) continue;

                        File outFile = new File(driverDir, new File(name).getName());
                        if (name.endsWith("meta.json")) {
                            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                            int read;
                            while ((read = zis.read(buffer)) != -1) baos.write(buffer, 0, read);
                            metaJson = baos.toString("UTF-8");
                            Files.write(outFile.toPath(), baos.toByteArray());
                            extractedFiles.add(outFile.getName());
                        } else if (name.endsWith(".so")) {
                            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(outFile)) {
                                int read;
                                while ((read = zis.read(buffer)) != -1) fos.write(buffer, 0, read);
                            }
                            extractedFiles.add(outFile.getName());
                        }
                    }
                }

                String driverTitle = "Driver Personalizado";
                boolean isTurnip = false;
                boolean isPanvk = false;

                if (metaJson != null) {
                    try {
                        org.json.JSONObject obj = new org.json.JSONObject(metaJson);
                        if (obj.has("name")) driverTitle = obj.getString("name");
                        if (obj.has("driverVersion")) driverTitle += " (" + obj.getString("driverVersion") + ")";
                        String jsonLower = metaJson.toLowerCase();
                        if (jsonLower.contains("turnip") || jsonLower.contains("freedreno") || jsonLower.contains("adreno")) isTurnip = true;
                        if (jsonLower.contains("panvk") || jsonLower.contains("panfrost")) isPanvk = true;
                    } catch (Exception ignored) {}
                }

                for (String f : extractedFiles) {
                    String fl = f.toLowerCase();
                    if (fl.contains("freedreno") || fl.contains("turnip") || fl.contains("adreno")) isTurnip = true;
                    if (fl.contains("panvk") || fl.contains("panfrost")) isPanvk = true;
                }

                final String finalTitle = driverTitle;
                final boolean finalIsTurnip = isTurnip;
                final boolean finalIsPanvk = isPanvk;
                final int fileCount = extractedFiles.size();

                writeSettingString("custom_driver_name", finalTitle);
                writeSetting("custom_driver_enabled", 1);

                runOnUiThread(() -> {
                    recreateSettings();
                    AlertDialog.Builder b = new AlertDialog.Builder(this);
                    b.setTitle("Driver Importado");
                    StringBuilder msg = new StringBuilder();
                    msg.append("Driver: ").append(finalTitle).append("\n");
                    msg.append("Arquivos registrados: ").append(fileCount).append("\n\n");

                    if (finalIsTurnip) {
                        msg.append("⚠️ AVISO DE COMPATIBILIDADE:\n")
                           .append("Este pacote é um driver Turnip desenvolvido exclusivamente para GPUs Qualcomm Adreno (/dev/kgsl-3d0).\n\n")
                           .append("O seu dispositivo possui SoC Exynos 1380 com GPU ARM Mali-G68. Hardware Mali não executa drivers Adreno. Para manter a estabilidade da corrida, o sistema utilizará o Vulkan Mali nativo.");
                    } else if (finalIsPanvk) {
                        msg.append("ℹ️ DRIVER PANVK (MALI):\n")
                           .append("Driver PanVK para Mali detectado e preparado. Requer suporte correspondente de kernel (/dev/mali0) no Android.");
                    } else {
                        msg.append("Driver customizado registrado com sucesso em dados do aplicativo.");
                    }
                    b.setMessage(msg.toString());
                    b.setPositiveButton("OK", null);
                    b.show();
                });
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, "Erro ao importar ZIP: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    private void addValueRow(String label, String value) {
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        TextView name = label(label, 10, 0xFFE4E7EB, false);
        row.addView(name, new LinearLayout.LayoutParams(0, dp(22), 1));
        TextView current = label(value, 10, WHITE, true);
        current.setGravity(Gravity.CENTER);
        current.setPadding(dp(8), 0, dp(8), 0);
        current.setBackground(outlined(0x33111519, 0x77FFFFFF, 6));
        row.addView(current, new LinearLayout.LayoutParams(-2, dp(22)));
        settingsPanel.addView(row);
    }

    private void toggleSettings() {
        if (settingsCard == null) return;
        settingsCard.setVisibility(settingsCard.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
    }

    private void pickIso() {
        if (importer.pick()) return;
        Toast.makeText(this, "Não foi possível abrir o seletor de arquivos.", Toast.LENGTH_LONG).show();
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == GameImporter.REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            File own = getExternalFilesDir(null);
            if (own != null) importer.copy(data.getData(), new File(own, "assets"), ok -> {
                if (ok) Toast.makeText(this, "Imagem pronta. Toque em JOGAR para iniciar.", Toast.LENGTH_LONG).show();
            });
        } else if (requestCode == REQUEST_PICK_DRIVER && resultCode == RESULT_OK && data != null && data.getData() != null) {
            importDriverZip(data.getData());
        }
    }

    private void addSpinner(String[] choices, int[] values, int current, String key) {
        int selected = 0;
        for (int i = 0; i < values.length; i++) if (values[i] == current) selected = i;
        final int[] selectedItem = {selected};
        FrameLayout choice = new FrameLayout(this);
        choice.setBackground(outlined(0x33111519, 0x66FFFFFF, 6));
        TextView value = label(choices[selected], 10, WHITE, false);
        value.setSingleLine(true);
        value.setGravity(Gravity.CENTER_VERTICAL);
        value.setPadding(dp(7), 0, dp(24), 0);
        choice.addView(value, new FrameLayout.LayoutParams(-1, -1));
        TextView arrow = label("⌄", 16, WHITE, true);
        arrow.setGravity(Gravity.CENTER);
        FrameLayout.LayoutParams arrowParams = new FrameLayout.LayoutParams(dp(28), -1, Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        choice.addView(arrow, arrowParams);
        choice.setClickable(true);
        choice.setFocusable(true);
        choice.setOnClickListener(v -> new AlertDialog.Builder(this)
                .setTitle("Selecionar opção")
                .setSingleChoiceItems(choices, selectedItem[0], (dialog, which) -> {
                    selectedItem[0] = which;
                    value.setText(choices[which]);
                    if (key != null) writeSetting(key, values[which]);
                    dialog.dismiss();
                }).setNegativeButton("Cancelar", null).show());
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, dp(25));
        params.bottomMargin = dp(1);
        settingsPanel.addView(choice, params);
    }

    private int readInt(String key, int fallback) {
        File own = getExternalFilesDir(null);
        if (own == null) return fallback;
        try {
            Matcher m = Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*=\\s*(-?\\d+)").matcher(
                    new String(Files.readAllBytes(new File(own, "forza.toml").toPath()), StandardCharsets.UTF_8));
            return m.find() ? Integer.parseInt(m.group(1)) : fallback;
        } catch (Exception ignored) { return fallback; }
    }

    private int readPercent(String key, int fallback) {
        File own = getExternalFilesDir(null);
        if (own == null) return fallback;
        try {
            Matcher m = Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*=\\s*(-?[0-9.]+)").matcher(
                    new String(Files.readAllBytes(new File(own, "forza.toml").toPath()), StandardCharsets.UTF_8));
            return m.find() ? Math.max(0, Math.min(100, Math.round(Float.parseFloat(m.group(1)) * 100))) : fallback;
        } catch (Exception ignored) { return fallback; }
    }

    private void writeSetting(String key, int value) {
        File own = getExternalFilesDir(null);
        if (own == null) return;
        try {
            own.mkdirs();
            File file = new File(own, "forza.toml");
            String content = file.isFile() ? new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8) : "";
            Pattern p = Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*=\\s*[^\\r\\n]*");
            Matcher m = p.matcher(content);
            String line = key + " = " + (key.equals("force_feedback") ? String.format(java.util.Locale.ROOT, "%.2f", value / 100.0) : value);
            content = m.find() ? m.replaceFirst(Matcher.quoteReplacement(line)) : content + (content.endsWith("\n") || content.isEmpty() ? "" : "\n") + line + "\n";
            File temp = new File(own, "forza.toml.tmp");
            Files.write(temp.toPath(), content.getBytes(StandardCharsets.UTF_8));
            if (!temp.renameTo(file)) {
                Files.copy(temp.toPath(), file.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                temp.delete();
            }
        } catch (Exception e) { Toast.makeText(this, "Não foi possível salvar a configuração.", Toast.LENGTH_SHORT).show(); }
    }

    private String readString(String key, String fallback) {
        File own = getExternalFilesDir(null);
        if (own == null) return fallback;
        try {
            Matcher m = Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*=\\s*\"([^\"]*)\"").matcher(
                    new String(Files.readAllBytes(new File(own, "forza.toml").toPath()), StandardCharsets.UTF_8));
            return m.find() ? m.group(1) : fallback;
        } catch (Exception ignored) { return fallback; }
    }

    private void writeSettingString(String key, String value) {
        File own = getExternalFilesDir(null);
        if (own == null) return;
        try {
            own.mkdirs();
            File file = new File(own, "forza.toml");
            String content = file.isFile() ? new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8) : "";
            Pattern p = Pattern.compile("(?m)^\\s*" + Pattern.quote(key) + "\\s*=\\s*[^\\r\\n]*");
            Matcher m = p.matcher(content);
            String line = key + " = \"" + value + "\"";
            content = m.find() ? m.replaceFirst(Matcher.quoteReplacement(line)) : content + (content.endsWith("\n") || content.isEmpty() ? "" : "\n") + line + "\n";
            File temp = new File(own, "forza.toml.tmp");
            Files.write(temp.toPath(), content.getBytes(StandardCharsets.UTF_8));
            if (!temp.renameTo(file)) {
                Files.copy(temp.toPath(), file.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                temp.delete();
            }
        } catch (Exception e) { Toast.makeText(this, "Não foi possível salvar a configuração.", Toast.LENGTH_SHORT).show(); }
    }

    private TextView section(String value) {
        TextView v = label(value, 10, 0xFFE3E5E8, true);
        v.setPadding(0, dp(2), 0, 0);
        return v;
    }
    private TextView label(String text, int size, int color, boolean bold) {
        TextView v = new TextView(this);
        v.setText(text); v.setTextColor(color); v.setTextSize(size);
        if (bold) v.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return v;
    }
    private GradientDrawable round(int color, int radius) {
        GradientDrawable d = new GradientDrawable(); d.setColor(color); d.setCornerRadius(dp(radius)); return d;
    }
    private GradientDrawable outlined(int color, int stroke, int radius) {
        GradientDrawable d = round(color, radius);
        d.setStroke(dp(1), stroke);
        return d;
    }
    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

}
