from distutils.core import setup
setup(
  name = 'bioval',         # How you named your package folder (MyLib)
  packages = ['bioval'],   # Chose the same as "name"
  version = '0.1.1.2-alpha',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Bioval is a Python package made to provide a wrapper and an easy access to a collection of evaluation metrics for comparing the similarity of two tensors, adapted to different evaluation processes of generative models applied to biological images, and by extension to natural images.',   # Give a short description about your library
  author = 'Ihab Bendidi, Ethan Cohen, Auguste Genovesio',                   # Type in your name
  author_email = 'bendidiihab@gmail.Com',      # Type in your E-Mail
  url = 'https://github.com/IhabBendidi/bioval',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/IhabBendidi/bioval/archive/refs/tags/v0.1.1-alpha.tar.gz',    # I explain this later on
  keywords = ['biology', 'evaluation', 'generative models',],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
        'numpy',
        'torch',
        'torchvision',
        'Pillow',
        'nvidia-ml-py',
        'piq',
        'pot',
        'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

