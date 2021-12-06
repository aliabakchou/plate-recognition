#include<iostream>
#include<stdlib.h>
#include<opencv2\objdetect.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\face.hpp>
#include<opencv2\face\facerec.hpp>
#include<filesystem>
#include<opencv2\imgcodecs.hpp>

using namespace cv::face;
using namespace std; //Pour pas avoir à faire std::cout par exemple
using namespace cv; // open CV pour pas avoir à faire cv::xxxx
namespace fs = std::filesystem;

string DATA = "./data/"; // data source folder
string MODELS = DATA + "Models/Source/"; // Contains the signatures source folders
string SIGNATURES = DATA + "Models/Dest/"; // Contains the collated signatures for the trainer
string CASCADE = MODELS + "haarcascade.xml"; // The default frontalface haarcascade features file provided by OpenCV
string PHOTOS = DATA + "Photos/"; // The folder containing the samples. Not used.
string EIGEN = DATA + "eigen.yml"; // The file to store the training data into

CascadeClassifier cascade;

// This sets the minimal size of our rectangles of interest.
// It prevents the model from wrongly detecting objects,
// but at the cost of missing objects that would be too small.
int side = 128;
Size SIZE = Size(side, side);

// Déclaration des fonctions

static void labelize(vector<Mat>& images, vector<int>& labels) {
	cout << "[FaceRecognizer] Creating a label for all the extracted model faces" << endl;

	vector<String> modelDir;

	// Stores all the subsequent files names in the cropped models directory.
	// Each folder being a label ranging from 1 to 4
	// The files (our training images) are named like so : label-index.jpg
	// (label and index being numbers)
	glob(SIGNATURES, modelDir, false);

	// For each file
	for (size_t i = 0; i < modelDir.size(); i++)
	{
		string name = "";
		// We extract the label from the image's name
		// Example : data/Models/Dest/12-4.jpg (so the 4th image previously found in the folder named "12")
		// We want the label ID ("12). In our string, it goes from the last path separator ('\') to the last '-'.
		// So from character 16 to character 18.
		// We then need a subsequence between these position to get our label
		size_t nameIndexStart = modelDir[i].rfind('\\', modelDir[i].length());
		size_t nameIndexEnd = modelDir[i].find_last_of('-', modelDir[i].length());
		if (nameIndexStart != string::npos)
		{
			name = modelDir[i].substr(nameIndexStart + 1, nameIndexEnd - nameIndexStart - 1);
		}
		// Both vectors end up with exactly the same size
		// Each image has the same position index as its related label, so the trainer knows how to name our objects.
		images.push_back(imread(modelDir[i], 0));
		labels.push_back(atoi(name.c_str()));
	}
}

void faceRecognitionTrainer() {
	vector<Mat> images;
	vector<int> labels;
	labelize(images, labels);

	cout << "[FaceRecognizer] Starting the Eigen Face Recognizer training" << endl;

	Ptr<EigenFaceRecognizer> eigen = EigenFaceRecognizer::create();

	eigen->train(images, labels);

	eigen->save(EIGEN);

	cout << "[FaceRecognizer] Training complete. Results have been stored in " << EIGEN << endl;
}

/*
 * Iterate over the models folder and assign a label for each of them.
 * The program basically just crops the faces in the images and give them a name with the following syntax:
 * <label>-<index>.jpg
 * label being the folder's reference (1 to 4 to n)
 * index being the image position in this folder (eg: 1-5 is the 5th signature image of the 1st object)
 */
void createSignatures()
{
	cout << "[FaceRecognizer] Starting the creation of the faces' signatures" << endl;

	vector<Rect> faces;
	string path;
	string name;
	Mat frame;
	Mat gray;
	Mat crop;
	Mat res;

	int count;
	int i;

	path = MODELS;

	if (fs::exists(path))
	{
		for (const auto& id : fs::directory_iterator(path))
		{
			name = id.path().filename().string();
			if (!name.ends_with(".xml"))
			{
				cout << "[FaceRecognizer] Parsing the faces in the '" << name << "' directory." << endl;
				count = 0;
				for (const auto& image : fs::directory_iterator(id.path()))
				{
					frame = imread(image.path().string());
					cvtColor(frame, gray, COLOR_BGR2GRAY);
					equalizeHist(gray, gray);

					cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, SIZE);

					Rect faceSignature;

					for (i = 0; i < faces.size(); i++)
					{
						faceSignature.x = faces[i].x;
						faceSignature.y = faces[i].y;
						faceSignature.width = (faces[i].width);
						faceSignature.height = (faces[i].height);

						crop = frame(faceSignature);
						resize(crop, res, SIZE, 0, 0, INTER_LINEAR);
						cvtColor(crop, gray, COLOR_BGR2GRAY);

						// Saves the cropped result into the model destination folder.
						imwrite(SIGNATURES + name + "-" + to_string(count++) + ".jpg", res);
					}
				}
			}
		}
		faceRecognitionTrainer();
	}
}
/* Fonction permettant le chargement du visage nécessaire à la détection d'objects
   Argument: objet CascadeClassifier qui pour nous va correspondre à un visage , et le chemin vers le fichier
   Valeur de retour : bool (true ou false) */
bool chargementFichier(CascadeClassifier &Visage, string filename) /*"&" Parce qu'on réutilise Visage donc on a besoin du contenu de l'adresse
si fichier .xml chargé true sinon false*/
{
	return Visage.load(_MACRO + filename) ? true : false;
}

/* Fonction demandant à l'utilisateur de saisir un chemin vers l'image
   Argument: un chemin "path" de type tableau de caractères, et une image de type Mat
   Valeur de retour : bool (true ou false) */
bool saisieImage(char path[150], Mat &img) {
	cout << "Veuillez saisir le chemin vers l'image que vous voulez analyser : \n";
	cin.getline(path, 100);
	img = imread(path, IMREAD_UNCHANGED);//flag pour charger l'image telle quelle
	return img.empty() ? true : false;//contenu vide ou pas
}

/* Fonction permettant de vérifier la saisie de l'utilisateur
   Argument: un chemin "path" de type tableau de caractères, et une image de type Mat
   Valeur de retour : aucune */
void VerifSaisie(char path[150], Mat &img)
{
	if (saisieImage(path, img))
	{
		while (img.empty())
		{
			cout << "Le chemin saisie n'est pas correct \n";
			cout << "Veuillez saisir le chemin vers l'image que vous voulez analyser : \n";
			cin.getline(path, 100);
			img = imread(path, IMREAD_UNCHANGED);
		}
	}
}

/* Fonction permettant de d'encadrer les visages détectés , et affiche le nombre de visages pr�sents
   Argument: une image de type Mat, vector<Rect> visages   A EXPLIQUER !!!!
   Valeur de retour : aucune */
/*
center = Point(faces[i].tl().x + faces[i].width / 2, faces[i].tl().y + faces[i].height / 2);
circle(original, center, faces[i].width / 2, Scalar(0, 255, 0), 2);
putText(original, name, Point(faces[i].tl().x + faces[i].width * 0.75, faces[i].tl().y + faces[i].height), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 0), 1);
*/

void drawVisage(Mat &img, vector<Rect> visages) {//vector = tableau d'objet (objet rectangle) dessine rectangle
	int i;
	Point center;
	for (i = 0; i < visages.size(); i++)
	{
		center = Point(visages[i].tl().x + visages[i].width / 2, visages[i].tl().y + visages[i].height / 2);
		Point p1(visages[i].x, visages[i].y);
		int h = (visages[i].x + visages[i].height);
		int w = (visages[i].y + visages[i].width);
		Point p2(h, w);
		//rectangle(img, p1, p2, Scalar(0, 255, 0), 2); //scalar = couleur BlueGreenRed épaisseur 2
		circle(img, center, visages[i].width / 2, Scalar(0, 255, 0), 2);
	}
	cout << i << " visage(s) detecte(s)\n"; //i = visage
	// ajouter le nom de la personne si reconnue
}


int main() {
	// Déclaration des variables
	CascadeClassifier Visage;
	char path[150];
	Mat img;
	vector<Rect> visages;

	long count = 0;
	int reco = 0;
	string name = "";
	
	cascade.load(CASCADE);
	//cascade.load("C:\\Users\\moi\\Desktop\\Projet\\Projet\\ReconnaissanceFaciale\\data\\Models\\Source\\haarcascade.xml");
	//cascade.load("./data/Models/Source/haarcascade.xml");

	if (cascade.empty())
	{
		cout << "Failed to open the Haar Cascade file at " << CASCADE << endl;
		return EXIT_FAILURE;
	}
	createSignatures();

	// Chargement du fichier haarcascade
	if (!chargementFichier(Visage, "\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"))
	{
		cout << "Le fichier n'a pas ete charge. \n";
		exit(1);
	}

	// Saisie de l'utilisateur
	VerifSaisie(path, img);

	cout << "L'image est chargee.\n";
	cout << "Veuillez patientez ...\n";

	// Détection des visages

	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();

	model->read(EIGEN);


	Mat frame;
	Mat grayImage;
	Mat original;

	original = frame.clone();

	// Convert the image to gray to get rid of unwanted details
	cvtColor(original, grayImage, COLOR_BGR2GRAY);
	equalizeHist(grayImage, grayImage);
	cascade.detectMultiScale(img, visages, 1.25);

	for (int i = 0; i < visages.size(); i++)
	{
		// region of interest containing the face
		Rect faceROI = visages[i];

		// crop the gray image to extract the face only
		Mat face = img(faceROI);

		// resizing the cropped image to fit the database image sizes
		Mat resized_face;
		resize(face, resized_face, SIZE, 1.0, 1.0, INTER_CUBIC);

		// Call to the Fisher's prediction method
		int label;
		double confidence;
		model->predict(resized_face, label, confidence);

		string text = name;

		// this sets a minimum threshold to prevent the model from wrongly assigning a face to the closest signature
		if (confidence < 3000) label = 0;

		// counts the recognized faces
		if (label) reco++;

		// Not the most optimized way, but surely the easiest one that worked when trying to associate a label number to a name
		switch (label) {
		case 0:
			name = "Unknown";
			break;
		case 1:
			name = "Rachel";
			break;
		case 2:
			name = "Joey";
			break;
		case 3:
			name = "Monica";
			break;
		case 4:
			name = "Chandler";
			break;
		}
		cout << "[FaceRecognizer] Confidence ratio: " << confidence << " - Signature label: " << label << " - Name: " << name << endl;
	}

	
	// Encadrage des visages et affichage de l'image
//	drawVisage(img, visages);
//	imshow("Affichage de l'image", img);
//	waitKey();
	return 0;
}  