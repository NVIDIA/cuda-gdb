unsigned int buffer[2] = {0, 2};

static int
func1 (void)
{
  buffer[0] = 1;
  asm volatile ("func1_label: .global func1_label\n");
  return 1;
}

static int
func2 (void)
{
  buffer[0] = 0;
  asm volatile ("func2_label: .global func2_label\n");
  return 2;
}

int
main (void)
{
  buffer[0] = 0;
  asm volatile ("main_label: .global main_label\n");
  func1 ();
  func2 ();
  return 0;
}