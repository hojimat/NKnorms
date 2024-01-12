import logging
import json
import itertools
from pylatex import Document, Figure, SubFigure, NoEscape

def create_figure(doc_name, fig_name: str, subfig_names: list, num_columns: int):
    """Create figure with given figure name and subfigure names and document object"""

    with doc_name.create(Figure(position='!htbp')) as figure:
        figure.add_caption(fig_name)


        for i,subfig in enumerate(subfig_names):
            with figure.create(SubFigure(position='b', width=NoEscape(rf"{1/num_columns}\linewidth"))) as subfigure:
                subfigure.add_image(subfig, width=NoEscape(r'\linewidth'))
                subfigure.add_caption(subfig.replace('../fig/','').replace('.pdf','').replace(fig_name,''))
            
            if i % num_columns == num_columns-1:
                figure.append(NoEscape(r'\\'))


def texify(dirs):
    """Generate proper file names and create figures for all but last 2 subdirs"""

    doc = Document()

    for combination in itertools.product(*dirs[:-2]):
        figure_name = ''.join(combination)
        subfigure_names = [f"../fig/{''.join(combination)}{row}{col}.pdf" for row, col in itertools.product(*dirs[-2:])]

        ncol = len(dirs[:-2])
        create_figure(doc, figure_name, subfigure_names, ncol)
        
    # Generate the document
    doc.generate_pdf('tex/sandbox', clean_tex=False)


def main():
    """Main function"""

    logging.basicConfig(level=logging.INFO)
    with open('structure.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  
        structure = data['structure']

    texify(structure)
    

if __name__=='__main__':
    main()
