import { ArrowRight, ArrowUpRight } from 'lucide-react'
import { MaskedLines, Reveal } from './reveal'
import { Link } from './router'

/* "How it works": the aim, the data, how the models were trained and what they scored, in plain words.
   Every number comes from final.ipynb (section in brackets); technical names are given next to the plain ones. */

const NOPRICE_SCORE = 0.950   // final.ipynb section 8, "type without the price"

const DATA = [
  { fig: '3,961', unit: 'listings', name: 'mebel.kz, Almaty', when: 'October 2026',
    use: 'Main data. Sizes, materials, colour, type, product name and price of each listing. Trains the price models and the type models.',
    rights: 'Public product pages. We keep the measured facts and the short product name; photos and page text are not used, because the store’s copyright notice forbids reusing them.' },
  { fig: '3,371', unit: 'listings · CC BY 4.0', name: 'Amazon Berkeley Objects', when: 'released 2022',
    use: 'Catalogue photos of Amazon furniture, with sizes. Trains the photo models (mebel.kz photos may not be used).',
    rights: 'Open licence CC BY 4.0. Credit: Amazon.com; dataset by Guillaumin et al. and Collins & Malik (UC Berkeley). No prices.',
    link: 'https://amazon-berkeley-objects.s3.amazonaws.com/index.html' },
  { fig: '620', unit: 'items · SAR', name: 'IKEA Saudi Arabia', when: 'April 2020',
    use: 'A second market. Checks whether the product name helps the price there too.',
    rights: 'Public listings collected by A. Kallam (Kaggle). No licence stated, so we use only facts (price, sizes, type) and the product label.' },
  { fig: '221', unit: 'photos · tested only', name: 'IKEA catalogue photos', when: 'Tautkute et al., 2017',
    use: 'A surprise test: photo models trained on Amazon are shown IKEA photos they have never seen.',
    rights: 'Property of IKEA.com, shared by the dataset authors for non-commercial use. Only tested on, never trained on, and not redistributed.' },
]

const STEPS = [
  ['Put an exam aside', 'Before any model is built, 793 mebel.kz listings are locked away as the final exam (the test set). No setting is ever chosen by looking at them.'],
  ['Keep twins together', 'One sofa design is often sold in six colours at the same price. All colours of one design stay on the same side, so a model cannot “recognise” a twin it has already seen.'],
  ['Practise in rounds', 'The other listings are cut into 5 parts. A model learns from 4 parts and is checked on the 5th, five times in turn (cross-validation). Its settings are chosen by the average of these checks.'],
  ['Pick, then sit the exam once', 'The model with the best cross-validation average is chosen and run once on the 793 exam listings. Both numbers are reported.'],
]

type Row = { task: string; model: string; plain: string; score: string; tech: string }
const MODELS: Row[] = [
  { task: 'Type, from sizes + materials + price', model: 'SVM (support vector machine)', plain: 'Draws flexible borders between the nine types in the space of sizes, materials and price.',
    score: '95 out of 100', tech: 'grouped CV macro-F1 0.949' },
  { task: 'Type, without the price', model: 'SVM, no price input', plain: 'The same, using only sizes, materials and colour.',
    score: `${Math.round(NOPRICE_SCORE * 100)} out of 100`, tech: `grouped CV macro-F1 ${NOPRICE_SCORE.toFixed(3)}` },
  { task: 'Type, from a photo', model: 'DINOv2 + logistic regression', plain: 'DINOv2 (Meta) already learned what objects look like from 142 million photos. We turn each photo into its 384 numbers and train a simple classifier on 3,332 Amazon listings.',
    score: '84 out of 100 (85 with sizes)', tech: 'grouped CV macro-F1 0.844 / 0.851' },
  { task: 'Price, from the specifications', model: 'LightGBM (boosted decision trees)', plain: 'Hundreds of small yes/no trees (“wider than 200 cm? fabric?”), each fixing the errors of the ones before.',
    score: '± 60 600 ₸ typical error (about 25%)', tech: 'CV MAE 60,591 KZT, MAPE 24.9%; exam 53,666 KZT' },
  { task: 'Price, specifications + product name', model: 'SVR on TF-IDF word pieces', plain: 'Reads the name as word pieces (“велюр”, “купе”, “подвесн”) and weighs them together with the sizes.',
    score: '± 59 100 ₸ typical error (about 23%)', tech: 'CV MAE 59,071 KZT, MAPE 23.0%; exam 51,746 KZT, 20.6%' },
]

const TRIED = [
  ['ResNet-18, frozen (2015, 1.3 M labelled photos)', '77'],
  ['Small network we trained from scratch', '61'],
  ['ResNet-18, fine-tuned on our photos', '79'],
  ['CLIP (OpenAI, 400 M captioned photos)', '80'],
  ['DINOv2 (Meta, 142 M photos), used in the app', '84'],
]

const FINDINGS = [
  ['The type is mostly a question of size.', 'From width, depth and height alone a model gets 92 out of 100 on the exam. Materials add the last few points, mostly for chests of drawers, TV stands and desks. The price adds nothing: without it the model scores the same (95 out of 100), so the app does not need it.'],
  ['Specifications price furniture to within about a quarter.', 'Two sofas of the same size can differ three times in price: fabric, mechanism and brand line are not in the size table.'],
  ['The product name helps a little, in two markets.', 'At mebel.kz the typical error falls from about 25% to 23% (21% on the exam). At IKEA Saudi Arabia, from 47% to 40%. The gain comes from words like велюр, шкаф-купе, подвесная, not from memorising design names.'],
  ['A photo sees what sizes cannot.', 'On Amazon, sizes alone find 8 of 100 TV stands; with the photo, 78. The best photo model also recognised IKEA photos it had never seen (88 out of 100).'],
]

const LIMITS = [
  'One store, one city, prices from one day. The price models do not know other shops, discounts on other days, or inflation.',
  'The ranges are wide: 8 in 10 real prices fall between about −31% and +40% of the estimate. Treat the number as “typical for this description”, not as a valuation.',
  'The name model learned only from real mebel.kz names. Made-up words or a name of another furniture type are ignored and the app falls back to the specifications.',
  'The photo models learned from Amazon catalogue photos (one object, plain background). Room photos, several objects or unusual angles make them guess.',
]

const GLOSSARY = [
  ['Typical error (MAE)', 'The average gap between the estimated and the real price, in tenge.'],
  ['Error in percent (MAPE)', 'The same gap as a share of the real price, averaged over listings.'],
  ['Score out of 100 (macro-F1)', 'How well the type is recognised, averaged over the nine types so that rare types (TV stands) count as much as common ones (sofas).'],
  ['Cross-validation (CV)', 'Learning on 4 parts and checking on the 5th, in turn: a fair estimate on listings the model did not learn from.'],
  ['Pretrained, frozen', 'A network trained by others on millions of photos, used as it is to turn a photo into numbers. Only the small classifier on top is trained by us.'],
  ['TF-IDF', 'Counts which word pieces a name contains, giving rare, informative pieces more weight than common ones.'],
]

export default function HowPage() {
  return (
    <main className="page how">
      <section className="how-hero">
        <Reveal><p className="eyebrow light">How it works</p></Reveal>
        <MaskedLines as="h1" lines={['What this project', 'is for, and how.']} className="display" start={120} immediate />
        <Reveal delay={420}>
          <div className="how-aim">
            <p className="lede">
              <b>The aim:</b> a furniture listing in Almaty shows sizes, materials, a name and a price. Is that price typical for what is
              sold? And which of these details actually decide what a piece of furniture is and what it costs?
            </p>
            <p>
              We built models that <b>estimate the price</b> of a listing from its description and <b>recognise its type</b> from its
              sizes or a photo, and we tested honestly which information helps. A shopper can check whether a price is in the usual
              range; a seller can see what similar listings cost.
            </p>
            <Link to="/predict" className="glass-btn">Try it <ArrowRight size={17} /></Link>
          </div>
        </Reveal>
      </section>

      <section className="section how-block" id="how-data">
        <div className="section-head">
          <Reveal><p className="eyebrow">1 · The data</p></Reveal>
          <MaskedLines lines={['Four sources,', 'each with a job.']} className="title" />
        </div>
        <div className="cards cards-4">
          {DATA.map((d, i) => (
            <Reveal key={d.name} delay={i * 90}>
              <article className="card surface">
                <div className="card-fig"><b>{d.fig}</b><span>{d.unit}</span></div>
                <h3>{d.name}</h3>
                <p className="how-when">{d.when}</p>
                <p><b>Used for:</b> {d.use}</p>
                <p className="how-rights"><b>Rights:</b> {d.rights}</p>
                {d.link && <a href={d.link} target="_blank" rel="noreferrer">Dataset page <ArrowUpRight size={14} /></a>}
              </article>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section how-block">
        <div className="section-head">
          <Reveal><p className="eyebrow">2 · Training</p></Reveal>
          <MaskedLines lines={['A fair test,', 'in four steps.']} className="title" />
        </div>
        <ol className="how-steps">
          {STEPS.map(([title, text], i) => (
            <Reveal key={title} delay={i * 90}>
              <li className="surface"><span className="how-n">{String(i + 1).padStart(2, '0')}</span><h3>{title}</h3><p>{text}</p></li>
            </Reveal>
          ))}
        </ol>
      </section>

      <section className="section how-block">
        <div className="section-head">
          <Reveal><p className="eyebrow">3 · The models in the app</p></Reveal>
          <MaskedLines lines={['What answers you,', 'and how good it is.']} className="title" />
        </div>
        <Reveal>
          <div className="how-table-wrap">
            <table className="how-table">
              <thead><tr><th>Question</th><th>Model</th><th>How it works</th><th>How good, in plain words</th><th>Technical score</th></tr></thead>
              <tbody>
                {MODELS.map((m) => (
                  <tr key={m.task}><td>{m.task}</td><td>{m.model}</td><td>{m.plain}</td><td className="how-score">{m.score}</td><td className="how-tech">{m.tech}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Reveal>
        <Reveal delay={120}>
          <div className="how-tried surface">
            <h3>Photo models we compared (score out of 100 for the type, same Amazon listings)</h3>
            <ul>{TRIED.map(([name, score]) => <li key={name}><span>{name}</span><i style={{ width: `${score}%` }} /><b>{score}</b></li>)}</ul>
            <p>A bigger, better pretrained model helped more than training a weaker one further. Price models compared: median price of the type, ridge regression, random forest, gradient boosting, LightGBM, SVR, and a pretrained text encoder (multilingual-e5) for the names.</p>
          </div>
        </Reveal>
      </section>

      <section className="section how-block">
        <div className="section-head">
          <Reveal><p className="eyebrow">4 · What we found</p></Reveal>
          <MaskedLines lines={['Four findings,', 'and their limits.']} className="title" />
        </div>
        <div className="how-two">
          <div className="how-list">
            {FINDINGS.map(([title, text], i) => (
              <Reveal key={title} delay={i * 80}><div className="how-item"><h3>{title}</h3><p>{text}</p></div></Reveal>
            ))}
          </div>
          <Reveal delay={160}>
            <aside className="how-limits surface">
              <h3>Limits</h3>
              <ul>{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
            </aside>
          </Reveal>
        </div>
      </section>

      <section className="section how-block how-last">
        <div className="section-head">
          <Reveal><p className="eyebrow">5 · Words used on this site</p></Reveal>
          <MaskedLines lines={['A short', 'glossary.']} className="title" />
        </div>
        <div className="how-glossary">
          {GLOSSARY.map(([term, text], i) => (
            <Reveal key={term} delay={i * 60}><div className="surface"><h3 className="gl-term">{term}</h3><p className="gl-text">{text}</p></div></Reveal>
          ))}
        </div>
        <p className="how-source">All numbers come from <code>final.ipynb</code> in the project archive (sections 1, 5.6, 7, 7.4 and 8).</p>
      </section>
    </main>
  )
}
