# Add your window to the PlugIn menu here
# Dont forget to add your packet as well

if {1} {  #test here if your target is configured
  # Uncomment this when the PlugIn class is ready
  #package require PLUGIN 1.0
  package require RHABOUT 1.0
  $Menu menubar_add_menu_command Other "About Red Hat" \
    {ManagedWin::open RHAbout -transient} \
    -underline 0
  # To activate the PlugIn sample, uncomment the next line
  #set plugins_available 1
}
