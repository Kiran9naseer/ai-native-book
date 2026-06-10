import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

// Feature icons as SVG components
const ROS2Icon = () => (
  <svg viewBox="0 0 48 48" className={styles.featureIcon}>
    <defs>
      <linearGradient id="ros2Grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style={{ stopColor: '#00D4FF' }} />
        <stop offset="100%" style={{ stopColor: '#0066FF' }} />
      </linearGradient>
    </defs>
    <circle cx="24" cy="24" r="20" fill="none" stroke="url(#ros2Grad)" strokeWidth="2" />
    <circle cx="24" cy="24" r="8" fill="url(#ros2Grad)" opacity="0.3" />
    <circle cx="24" cy="24" r="4" fill="url(#ros2Grad)" />
    <circle cx="12" cy="18" r="3" fill="url(#ros2Grad)" />
    <circle cx="36" cy="18" r="3" fill="url(#ros2Grad)" />
    <circle cx="18" cy="36" r="3" fill="url(#ros2Grad)" />
    <circle cx="30" cy="36" r="3" fill="url(#ros2Grad)" />
    <line x1="24" y1="24" x2="12" y2="18" stroke="url(#ros2Grad)" strokeWidth="1.5" />
    <line x1="24" y1="24" x2="36" y2="18" stroke="url(#ros2Grad)" strokeWidth="1.5" />
    <line x1="24" y1="24" x2="18" y2="36" stroke="url(#ros2Grad)" strokeWidth="1.5" />
    <line x1="24" y1="24" x2="30" y2="36" stroke="url(#ros2Grad)" strokeWidth="1.5" />
  </svg>
);

const SimIcon = () => (
  <svg viewBox="0 0 48 48" className={styles.featureIcon}>
    <defs>
      <linearGradient id="simGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style={{ stopColor: '#8B5CF6' }} />
        <stop offset="100%" style={{ stopColor: '#6366F1' }} />
      </linearGradient>
    </defs>
    <rect x="6" y="10" width="36" height="24" rx="3" fill="none" stroke="url(#simGrad)" strokeWidth="2" />
    <rect x="10" y="14" width="28" height="16" rx="1" fill="url(#simGrad)" opacity="0.2" />
    <rect x="18" y="34" width="12" height="6" fill="url(#simGrad)" />
    <rect x="14" y="40" width="20" height="2" rx="1" fill="url(#simGrad)" />
    <circle cx="24" cy="22" r="6" fill="none" stroke="url(#simGrad)" strokeWidth="1.5" />
    <path d="M21 22 L27 22 M24 19 L24 25" stroke="url(#simGrad)" strokeWidth="1.5" />
  </svg>
);

const IsaacIcon = () => (
  <svg viewBox="0 0 48 48" className={styles.featureIcon}>
    <defs>
      <linearGradient id="isaacGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style={{ stopColor: '#10B981' }} />
        <stop offset="100%" style={{ stopColor: '#059669' }} />
      </linearGradient>
    </defs>
    <path d="M24 6 L24 42" stroke="url(#isaacGrad)" strokeWidth="2" strokeDasharray="4 2" />
    <circle cx="24" cy="16" r="8" fill="none" stroke="url(#isaacGrad)" strokeWidth="2" />
    <circle cx="24" cy="16" r="3" fill="url(#isaacGrad)" />
    <path d="M10 38 L24 28 L38 38" fill="none" stroke="url(#isaacGrad)" strokeWidth="2" />
    <circle cx="10" cy="38" r="3" fill="url(#isaacGrad)" />
    <circle cx="38" cy="38" r="3" fill="url(#isaacGrad)" />
    <circle cx="24" cy="28" r="3" fill="url(#isaacGrad)" />
  </svg>
);

const VLAIcon = () => (
  <svg viewBox="0 0 48 48" className={styles.featureIcon}>
    <defs>
      <linearGradient id="vlaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style={{ stopColor: '#F59E0B' }} />
        <stop offset="100%" style={{ stopColor: '#D97706' }} />
      </linearGradient>
    </defs>
    <circle cx="24" cy="18" r="10" fill="none" stroke="url(#vlaGrad)" strokeWidth="2" />
    <circle cx="20" cy="16" r="2" fill="url(#vlaGrad)" />
    <circle cx="28" cy="16" r="2" fill="url(#vlaGrad)" />
    <path d="M19 22 Q24 26 29 22" stroke="url(#vlaGrad)" strokeWidth="1.5" fill="none" />
    <path d="M12 32 L18 28 L24 32 L30 28 L36 32" stroke="url(#vlaGrad)" strokeWidth="2" fill="none" />
    <path d="M14 36 L18 34 L24 38 L30 34 L34 36" stroke="url(#vlaGrad)" strokeWidth="2" fill="none" opacity="0.6" />
    <path d="M16 40 L22 38 L26 40 L32 38" stroke="url(#vlaGrad)" strokeWidth="2" fill="none" opacity="0.3" />
  </svg>
);

const features = [
  {
    title: 'Module 1: ROS 2 Fundamentals',
    Icon: ROS2Icon,
    description: 'Master the Robot Operating System 2 — the industry-standard middleware for robotics. Learn nodes, topics, services, and actions.',
    link: '/docs/modules/ros2/introduction',
    color: '#00D4FF',
  },
  {
    title: 'Module 2: Gazebo & Unity Simulation',
    Icon: SimIcon,
    description: 'Build realistic simulation environments for humanoid robots. Test before deploying to physical hardware.',
    link: '/docs/modules/gazebo-unity/introduction',
    color: '#8B5CF6',
  },
  {
    title: 'Module 3: Isaac AI Brain',
    Icon: IsaacIcon,
    description: 'Leverage NVIDIA Isaac for advanced perception pipelines, visual SLAM, and Nav2 navigation systems.',
    link: '/docs/modules/isaac-ai-brain/introduction',
    color: '#10B981',
  },
  {
    title: 'Module 4: Vision-Language-Action',
    Icon: VLAIcon,
    description: 'Build voice-controlled humanoid robots with LLM-based cognitive planning and autonomous behavior.',
    link: '/docs/modules/vla/introduction',
    color: '#F59E0B',
  },
];

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className={styles.heroBackground}>
        <div className={styles.heroGlow}></div>
        <div className={styles.heroGrid}></div>
      </div>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroBadge}>
            <span className={styles.badgeIcon}>🤖</span>
            <span>Physical AI & Humanoid Robotics</span>
          </div>
          <Heading as="h1" className={styles.heroTitle}>
            AI-Native Book
          </Heading>
          <p className={styles.heroSubtitle}>
            The comprehensive guide to building intelligent humanoid robots.
            From ROS 2 fundamentals to vision-language-action models.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro">
              📚 Start Learning
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="/docs/modules/ros2/introduction">
              🚀 Quick Start
            </Link>
          </div>
          <div className={styles.heroStats}>
            <div className={styles.statItem}>
              <span className={styles.statNumber}>4</span>
              <span className={styles.statLabel}>Modules</span>
            </div>
            <div className={styles.statDivider}></div>
            <div className={styles.statItem}>
              <span className={styles.statNumber}>50+</span>
              <span className={styles.statLabel}>Tutorials</span>
            </div>
            <div className={styles.statDivider}></div>
            <div className={styles.statItem}>
              <span className={styles.statNumber}>100%</span>
              <span className={styles.statLabel}>Open Source</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({ title, Icon, description, link, color }) {
  return (
    <div className={clsx('col col--6', styles.featureCol)}>
      <Link to={link} className={styles.featureLink}>
        <div className={styles.featureCard} style={{ '--feature-color': color }}>
          <div className={styles.featureIconWrapper}>
            <Icon />
          </div>
          <div className={styles.featureContent}>
            <h3 className={styles.featureTitle}>{title}</h3>
            <p className={styles.featureDescription}>{description}</p>
            <span className={styles.featureArrow}>
              Explore →
            </span>
          </div>
        </div>
      </Link>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Learning Modules</h2>
          <p className={styles.sectionSubtitle}>
            Four comprehensive modules taking you from basics to advanced humanoid AI
          </p>
        </div>
        <div className="row">
          {features.map((props, idx) => (
            <FeatureCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function HomepageCTA() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaCard}>
          <div className={styles.ctaContent}>
            <h2 className={styles.ctaTitle}>Ready to Build the Future?</h2>
            <p className={styles.ctaDescription}>
              Join the community of developers building next-generation humanoid robots.
            </p>
            <div className={styles.ctaButtons}>
              <Link
                className="button button--primary button--lg"
                to="https://github.com/ai-native-book/ai-native-book">
                ⭐ Star on GitHub
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/blog">
                📝 Read the Blog
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="Welcome"
      description="Physical AI & Humanoid Robotics Documentation">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <HomepageCTA />
      </main>
    </Layout>
  );
}
