void test_solve_sec_ode(){
  double ode_args[][6] = {{+3212.1180920684, +20142049.7981415428, +0.2181661565, +0.0000000000, +0.0000000000, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1772799703, +0.1773946861, -0.3604152962, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1541355236, +0.1542330151, -0.3062107375, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1342977801, +0.1343814490, -0.2627969945, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.1672524423, -0.1565103287, -35.2768780619, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0530380959, -0.0530752473, +0.1093355263, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0315222668, -0.0316352884, +0.3550704837, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0073451360, -0.0074534165, +0.3399300283, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.0158688323, +0.0157655295, +0.3243099336, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.0380150140, +0.0379163957, +0.3096013601, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.0591643230, +0.0590701322, +0.2957019346, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.0793626427, +0.0792726958, +0.2823789341, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.0986490644, +0.0985631863, +0.2696054178, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1170614176, +0.1169794389, +0.2573641113, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1346362202, +0.1345579774, +0.2456355071, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1514086383, +0.1513339742, +0.2344005946, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1674125354, +0.1673412986, +0.2236410883, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1826805193, +0.1826125641, +0.2133393338, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.1972439846, +0.1971791704, +0.2034782590, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2111331517, +0.2110713435, +0.1940413395, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2243771043, +0.2243181722, +0.1850125697, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2370038251, +0.2369476438, +0.1763764394, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2490402289, +0.2489866783, +0.1681179138, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2605121963, +0.2604611606, +0.1602224170, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2714446042, +0.2713959724, +0.1526758184, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2818613575, +0.2818150228, +0.1454644208, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.2917854181, +0.2917412779, +0.1385749500, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.3012388350, +0.3011967909, +0.1319945464, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.3102427724, +0.3102027299, +0.1257107560, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.3188175380, +0.3187794064, +0.1197115236, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.3269826105, +0.3269463030, +0.1139851851, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.3347566671, +0.3347221002, +0.1085204609, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0000000000, +0.0000000000, +0.0000000000, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0027096985, -0.0027033161, -0.0201241688, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0038211831, -0.0038166905, -0.0141091798, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, -0.0047653041, -0.0047611348, -0.0130889736, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +4.4691470517, +4.2909292593, +577.4427244006, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.5594211765, +0.5575628555, +5.7965816797, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9107072727, +0.9104516346, +0.8051195660, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9390296551, +0.9389763114, +0.1684119021, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9473005011, +0.9472679888, +0.1020566569, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9544447716, +0.9544123380, +0.1018142668, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9614020123, +0.9613711776, +0.0968090169, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9679607713, +0.9679318070, +0.0909371018, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9741223079, +0.9740950878, +0.0854606728, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9799143679, +0.9798887750, +0.0803514956, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9853607943, +0.9853367261, +0.0755648531, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9904833254, +0.9904606859, +0.0710789104, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9953022996, +0.9952809993, +0.0668744552, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +0.9998367433, +0.9998166983, +0.0629331960, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0041044439, +1.0040855759, +0.0592379927, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0081220311, +1.0081042668, +0.0557728277, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0119050521, +1.0118883229, +0.0525227322, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0154680426, +1.0154522846, +0.0494737197, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0188245936, +1.0188097468, +0.0466127236, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0219874131, +1.0219734216, +0.0439275398, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0249683851, +1.0249551964, +0.0414067721, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0277786236, +1.0277661888, +0.0390397795, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0304285241, +1.0304167974, +0.0368166288, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0329278113, +1.0329167499, +0.0347280486, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0352855838, +1.0352751475, +0.0327653847, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0375103558, +1.0375005071, +0.0309205614, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0396100962, +1.0396007999, +0.0291860423, +0.0002500000},
                          {+3212.1180920684, +20142049.7981415428, +1.0415922650, +1.0415834884, +0.0275547947, +0.0002500000}};
  double ode_outs[][2] = {{+0.0967423486, +607.9612389744},
                          {+0.1772939531, -0.3601025017},
                          {+0.1541474192, -0.3060241844},
                          {+0.1343079889, -0.2626355946},
                          {-0.1661543699, -33.8917041728},
                          {-0.0530436463, +0.1157927631},
                          {-0.0315360461, +0.3547819099},
                          {-0.0073583713, +0.3398715729},
                          {+0.0158562064, +0.3242486473},
                          {+0.0380029603, +0.3095444692},
                          {+0.0591528105, +0.2956475290},
                          {+0.0793516490, +0.2823267552},
                          {+0.0986385680, +0.2695554053},
                          {+0.1170513979, +0.2573161886},
                          {+0.1346266571, +0.2455895965},
                          {+0.1513995126, +0.2343566216},
                          {+0.1674038286, +0.2235989810},
                          {+0.1826722136, +0.2132990228},
                          {+0.1972360628, +0.2034396772},
                          {+0.2111255973, +0.1940044219},
                          {+0.2243699015, +0.1849772532},
                          {+0.2369969585, +0.1763426630},
                          {+0.2490336839, +0.1680856182},
                          {+0.2605059586, +0.1601915449},
                          {+0.2714386604, +0.1526463141},
                          {+0.2818556944, +0.1454362303},
                          {+0.2917800233, +0.1385480212},
                          {+0.3012336964, +0.1319688286},
                          {+0.3102378784, +0.1256862002},
                          {+0.3188128776, +0.1196880823},
                          {+0.3269781731, +0.1139628124},
                          {+0.3347524423, +0.1084991126},
                          {+0.0000000000, +0.0000000000},
                          {-0.0027089305, -0.0200429745},
                          {-0.0038206347, -0.0141022359},
                          {-0.0047647945, -0.0130865190},
                          {+4.4498473807, +561.4055014284},
                          {+0.5591888663, +5.8287270645},
                          {+0.9106763831, +0.8026888685},
                          {+0.9390232660, +0.1675420266},
                          {+0.9472965256, +0.1020486032},
                          {+0.9544408063, +0.1018021615},
                          {+0.9613982444, +0.0967852738},
                          {+0.9679572321, +0.0909143549},
                          {+0.9741189818, +0.0854395510},
                          {+0.9799112405, +0.0803317236},
                          {+0.9853578533, +0.0755463234},
                          {+0.9904805589, +0.0710615444},
                          {+0.9952996968, +0.0668581777},
                          {+0.9998342938, +0.0629179361},
                          {+1.0041021383, +0.0592236841},
                          {+1.0081198604, +0.0557594085},
                          {+1.0119030078, +0.0525101447},
                          {+1.0154661170, +0.0494619097},
                          {+1.0188227793, +0.0466016405},
                          {+1.0219857033, +0.0439171366},
                          {+1.0249667734, +0.0413970046},
                          {+1.0277771040, +0.0390306068},
                          {+1.0304270911, +0.0368080124},
                          {+1.0329264596, +0.0347199527},
                          {+1.0352843085, +0.0327577759},
                          {+1.0375091522, +0.0309134085},
                          {+1.0396089601, +0.0291793161},
                          {+1.0415911925, +0.0275484682}};

  double a, b, c, y_init, ydot_init, t_f;
  double expected_y_tf, expected_ydot_tf;
  double* y_tf = (double *) malloc(sizeof(double));
  double* ydot_tf = (double *) malloc(sizeof(double));

  std::cout << "Starting ODE Tests" << std::endl;
  std::cout << "------------------" << std::endl;
  for (int i=0; i<64; i++){
    a = ode_args[i][0];
    b = ode_args[i][1];
    c = ode_args[i][2];
    y_init = ode_args[i][3];
    ydot_init = ode_args[i][4];
    t_f = ode_args[i][5];

    expected_y_tf = ode_outs[i][0];
    expected_ydot_tf = ode_outs[i][1];

    solve_sec_ode(a, b, c, y_init, ydot_init, t_f, y_tf, ydot_tf);
    std::cout << *y_tf    << " == " << expected_y_tf    << " , ";
    std::cout << *ydot_tf << " == " << expected_ydot_tf << std::endl;
  }
  std::cout << "------------------" << std::endl;

}

void solve_sec_ode(double a, double b, double c,
                   double y_init, double ydot_init,
                   double t_f, double* y_tf, double* ydot_tf){
  /*
  This function solves the homogeneous ODE with constant coefficients
  y_ddot + a * y_dot + b * y - b * c = 0
  The exact y function is computed and evaluated at the final time t_f.
  y_tf will be y(t_f).
  ydot_tf will be ydot(t_f).
  */

  double x_init = y_init - c;
  double delta = (a*a) - 4*b;
  double delta_sqrt, r1, r2, c1, c2, e1, e2, sin_, cos_;

  if (delta > 0) {
    // Two real roots case
    delta_sqrt = sqrt(delta);
    r1 = (- a - delta_sqrt) / 2;
    r2 = (- a + delta_sqrt) / 2;
    c1 = (x_init * r2 - ydot_init ) / delta_sqrt;
    c2 = (ydot_init - - x_init * r1) / delta_sqrt;
    e1 = exp(r1 * t_f);
    e2 = exp(r2 * t_f);
    *y_tf = c1 * e1 + c2 * e2 + c;
    *ydot_tf = c1 * r1 * e1 + c2 * r2 * e2;
  } else if (delta == 0){
    // One real root case
    r1 = -a / 2;
    c1 = x_init;
    c2 = ydot_init - c1 * r1;
    e1 = exp(r1 * t_f);
    *y_tf = e1 * (c1 + c2 * t_f) + c;
    *ydot_tf = e1 * (c1 * r1 + c2 * (1 + r1 * t_f));
  } else {
    // Two imaginary roots case
    r1 = -a / 2;
    r2 = sqrt(-delta) / 2;
    c1 = x_init;
    c2 = (ydot_init - c1 * r1) / r2;
    e1 = exp(r1 * t_f);
    cos_ = cos(r2 * t_f);
    sin_ = sin(r2 * t_f);
    *y_tf = e1 * (c1 * cos_ + c2 * sin_) + c;
    *ydot_tf = e1 * (c1 * (r1 * cos_ - r2 * sin_) + c2 * (r1 * sin_ + r2 *cos_));
  }
}

/////////////////////////////////////
void solve_motor_ode(double tau_cmd, double y_init, double ydot_init,
                     double t_f, double* y_tf, double* ydot_tf){
  /*
  This function solves the homogeneous ODE with constant coefficients
  y_ddot + 2 * zeta * omega0 * y_dot + (omega0 ** 2) * (y - tau_cmd) = 0
  The exact y function is computed and evaluated at the final time t_f.
  y_tf will be y(t_f).
  ydot_tf will be ydot(t_f).
  */
  double c1, c2, g, exp_, sin_, cos_;
  // Two imaginary roots case
  c1 = y_init - tau_cmd;
  c2 = (ydot_init - c1 * motor_root_real) / motor_root_imag;
  exp_ = exp(motor_root_real * t_f);
  g = motor_root_imag * t_f;
  cos_ = cos(g);
  sin_ = sin(g);
  *y_tf = exp_ * (c1 * cos_ + c2 * sin_) + tau_cmd;
  *ydot_tf = exp_ * (c1 * (motor_root_real * cos_ - motor_root_imag * sin_) +
                     c2 * (motor_root_real * sin_ + motor_root_imag * cos_));
}

void test_solve_motor_ode(){
  double ode_args[][4] = {{+0.2181661565, +0.0000000000, +0.0000000000, +0.0002500000},
                          {+0.1772799703, +0.1773946861, -0.3604152962, +0.0002500000},
                          {+0.1541355236, +0.1542330151, -0.3062107375, +0.0002500000},
                          {+0.1342977801, +0.1343814490, -0.2627969945, +0.0002500000},
                          {-0.1672524423, -0.1565103287, -35.2768780619, +0.0002500000},
                          {-0.0530380959, -0.0530752473, +0.1093355263, +0.0002500000},
                          {-0.0315222668, -0.0316352884, +0.3550704837, +0.0002500000},
                          {-0.0073451360, -0.0074534165, +0.3399300283, +0.0002500000},
                          {+0.0158688323, +0.0157655295, +0.3243099336, +0.0002500000},
                          {+0.0380150140, +0.0379163957, +0.3096013601, +0.0002500000},
                          {+0.0591643230, +0.0590701322, +0.2957019346, +0.0002500000},
                          {+0.0793626427, +0.0792726958, +0.2823789341, +0.0002500000},
                          {+0.0986490644, +0.0985631863, +0.2696054178, +0.0002500000},
                          {+0.1170614176, +0.1169794389, +0.2573641113, +0.0002500000},
                          {+0.1346362202, +0.1345579774, +0.2456355071, +0.0002500000},
                          {+0.1514086383, +0.1513339742, +0.2344005946, +0.0002500000},
                          {+0.1674125354, +0.1673412986, +0.2236410883, +0.0002500000},
                          {+0.1826805193, +0.1826125641, +0.2133393338, +0.0002500000},
                          {+0.1972439846, +0.1971791704, +0.2034782590, +0.0002500000},
                          {+0.2111331517, +0.2110713435, +0.1940413395, +0.0002500000},
                          {+0.2243771043, +0.2243181722, +0.1850125697, +0.0002500000},
                          {+0.2370038251, +0.2369476438, +0.1763764394, +0.0002500000},
                          {+0.2490402289, +0.2489866783, +0.1681179138, +0.0002500000},
                          {+0.2605121963, +0.2604611606, +0.1602224170, +0.0002500000},
                          {+0.2714446042, +0.2713959724, +0.1526758184, +0.0002500000},
                          {+0.2818613575, +0.2818150228, +0.1454644208, +0.0002500000},
                          {+0.2917854181, +0.2917412779, +0.1385749500, +0.0002500000},
                          {+0.3012388350, +0.3011967909, +0.1319945464, +0.0002500000},
                          {+0.3102427724, +0.3102027299, +0.1257107560, +0.0002500000},
                          {+0.3188175380, +0.3187794064, +0.1197115236, +0.0002500000},
                          {+0.3269826105, +0.3269463030, +0.1139851851, +0.0002500000},
                          {+0.3347566671, +0.3347221002, +0.1085204609, +0.0002500000},
                          {-0.0000000000, +0.0000000000, +0.0000000000, +0.0002500000},
                          {-0.0027096985, -0.0027033161, -0.0201241688, +0.0002500000},
                          {-0.0038211831, -0.0038166905, -0.0141091798, +0.0002500000},
                          {-0.0047653041, -0.0047611348, -0.0130889736, +0.0002500000},
                          {+4.4691470517, +4.2909292593, +577.4427244006, +0.0002500000},
                          {+0.5594211765, +0.5575628555, +5.7965816797, +0.0002500000},
                          {+0.9107072727, +0.9104516346, +0.8051195660, +0.0002500000},
                          {+0.9390296551, +0.9389763114, +0.1684119021, +0.0002500000},
                          {+0.9473005011, +0.9472679888, +0.1020566569, +0.0002500000},
                          {+0.9544447716, +0.9544123380, +0.1018142668, +0.0002500000},
                          {+0.9614020123, +0.9613711776, +0.0968090169, +0.0002500000},
                          {+0.9679607713, +0.9679318070, +0.0909371018, +0.0002500000},
                          {+0.9741223079, +0.9740950878, +0.0854606728, +0.0002500000},
                          {+0.9799143679, +0.9798887750, +0.0803514956, +0.0002500000},
                          {+0.9853607943, +0.9853367261, +0.0755648531, +0.0002500000},
                          {+0.9904833254, +0.9904606859, +0.0710789104, +0.0002500000},
                          {+0.9953022996, +0.9952809993, +0.0668744552, +0.0002500000},
                          {+0.9998367433, +0.9998166983, +0.0629331960, +0.0002500000},
                          {+1.0041044439, +1.0040855759, +0.0592379927, +0.0002500000},
                          {+1.0081220311, +1.0081042668, +0.0557728277, +0.0002500000},
                          {+1.0119050521, +1.0118883229, +0.0525227322, +0.0002500000},
                          {+1.0154680426, +1.0154522846, +0.0494737197, +0.0002500000},
                          {+1.0188245936, +1.0188097468, +0.0466127236, +0.0002500000},
                          {+1.0219874131, +1.0219734216, +0.0439275398, +0.0002500000},
                          {+1.0249683851, +1.0249551964, +0.0414067721, +0.0002500000},
                          {+1.0277786236, +1.0277661888, +0.0390397795, +0.0002500000},
                          {+1.0304285241, +1.0304167974, +0.0368166288, +0.0002500000},
                          {+1.0329278113, +1.0329167499, +0.0347280486, +0.0002500000},
                          {+1.0352855838, +1.0352751475, +0.0327653847, +0.0002500000},
                          {+1.0375103558, +1.0375005071, +0.0309205614, +0.0002500000},
                          {+1.0396100962, +1.0396007999, +0.0291860423, +0.0002500000},
                          {+1.0415922650, +1.0415834884, +0.0275547947, +0.0002500000}};
  double ode_outs[][2] = {{+0.0967423486, +607.9612389744},
                          {+0.1772939531, -0.3601025017},
                          {+0.1541474192, -0.3060241844},
                          {+0.1343079889, -0.2626355946},
                          {-0.1661543699, -33.8917041728},
                          {-0.0530436463, +0.1157927631},
                          {-0.0315360461, +0.3547819099},
                          {-0.0073583713, +0.3398715729},
                          {+0.0158562064, +0.3242486473},
                          {+0.0380029603, +0.3095444692},
                          {+0.0591528105, +0.2956475290},
                          {+0.0793516490, +0.2823267552},
                          {+0.0986385680, +0.2695554053},
                          {+0.1170513979, +0.2573161886},
                          {+0.1346266571, +0.2455895965},
                          {+0.1513995126, +0.2343566216},
                          {+0.1674038286, +0.2235989810},
                          {+0.1826722136, +0.2132990228},
                          {+0.1972360628, +0.2034396772},
                          {+0.2111255973, +0.1940044219},
                          {+0.2243699015, +0.1849772532},
                          {+0.2369969585, +0.1763426630},
                          {+0.2490336839, +0.1680856182},
                          {+0.2605059586, +0.1601915449},
                          {+0.2714386604, +0.1526463141},
                          {+0.2818556944, +0.1454362303},
                          {+0.2917800233, +0.1385480212},
                          {+0.3012336964, +0.1319688286},
                          {+0.3102378784, +0.1256862002},
                          {+0.3188128776, +0.1196880823},
                          {+0.3269781731, +0.1139628124},
                          {+0.3347524423, +0.1084991126},
                          {+0.0000000000, +0.0000000000},
                          {-0.0027089305, -0.0200429745},
                          {-0.0038206347, -0.0141022359},
                          {-0.0047647945, -0.0130865190},
                          {+4.4498473807, +561.4055014284},
                          {+0.5591888663, +5.8287270645},
                          {+0.9106763831, +0.8026888685},
                          {+0.9390232660, +0.1675420266},
                          {+0.9472965256, +0.1020486032},
                          {+0.9544408063, +0.1018021615},
                          {+0.9613982444, +0.0967852738},
                          {+0.9679572321, +0.0909143549},
                          {+0.9741189818, +0.0854395510},
                          {+0.9799112405, +0.0803317236},
                          {+0.9853578533, +0.0755463234},
                          {+0.9904805589, +0.0710615444},
                          {+0.9952996968, +0.0668581777},
                          {+0.9998342938, +0.0629179361},
                          {+1.0041021383, +0.0592236841},
                          {+1.0081198604, +0.0557594085},
                          {+1.0119030078, +0.0525101447},
                          {+1.0154661170, +0.0494619097},
                          {+1.0188227793, +0.0466016405},
                          {+1.0219857033, +0.0439171366},
                          {+1.0249667734, +0.0413970046},
                          {+1.0277771040, +0.0390306068},
                          {+1.0304270911, +0.0368080124},
                          {+1.0329264596, +0.0347199527},
                          {+1.0352843085, +0.0327577759},
                          {+1.0375091522, +0.0309134085},
                          {+1.0396089601, +0.0291793161},
                          {+1.0415911925, +0.0275484682}};

  double tau_cmd, y_init, ydot_init, t_f;
  double expected_y_tf, expected_ydot_tf;
  double* y_tf = (double *) malloc(sizeof(double));
  double* ydot_tf = (double *) malloc(sizeof(double));

  std::cout << "Starting ODE Tests" << std::endl;
  std::cout << "------------------" << std::endl;
  for (int i=0; i<64; i++){
    tau_cmd = ode_args[i][0];
    y_init = ode_args[i][1];
    ydot_init = ode_args[i][2];
    t_f = ode_args[i][3];

    expected_y_tf = ode_outs[i][0];
    expected_ydot_tf = ode_outs[i][1];

    solve_motor_ode(tau_cmd, y_init, ydot_init, t_f, y_tf, ydot_tf);
    std::cout << *y_tf    << " == " << expected_y_tf    << " , ";
    std::cout << *ydot_tf << " == " << expected_ydot_tf << std::endl;
  }
  std::cout << "------------------" << std::endl;

}
